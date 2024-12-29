import torch
from PIL import Image, ImageDraw
import cv2
import numpy as np
from nudenet import NudeDetector
from diffusers import StableDiffusionInpaintPipeline
import os
from tqdm import tqdm
import tempfile
import logging
from datetime import datetime

class EnhancedVideoClothingGenerator:
    def setup_logging(self):
        """Set up logging configuration"""
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f'video_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        self.logger = logging.getLogger('VideoClothingGenerator')
        self.logger.setLevel(logging.INFO)
        
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file)
        
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(log_format)
        f_handler.setFormatter(log_format)
        
        self.logger.handlers = []
        
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)
        
        self.logger.info('Logging initialized for video processing')

    def __init__(self):
        self.setup_logging()
        
        self.logger.info("Initializing NudeDetector...")
        self.nude_detector = NudeDetector()

        self.logger.info("Setting up Stable Diffusion pipeline...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
            self.logger.info("CUDA is available. Using GPU acceleration.")
        else:
            self.logger.info("CUDA not available. Using CPU mode.")

        self.target_classes = {
            "FEMALE_BREAST_EXPOSED": {
                "prompt": "wearing a fitted black athletic sports bra, athletic wear",
                "priority": 1
            },
            "FEMALE_BREAST_COVERED": {
                "prompt": "wearing a fitted black athletic sports bra, athletic wear",
                "priority": 1
            },
            "BUTTOCKS_EXPOSED": {
                "prompt": "wearing black athletic leggings, athletic wear",
                "priority": 2
            },
            "FEMALE_GENITALIA_EXPOSED": {
                "prompt": "wearing black athletic leggings, athletic wear",
                "priority": 2
            },
            "FEMALE_GENITALIA_COVERED": {
                "prompt": "wearing black athletic leggings, athletic wear",
                "priority": 2
            },
            "MALE_GENITALIA_EXPOSED": {
                "prompt": "wearing black athletic shorts, athletic wear",
                "priority": 2
            },
            "ANUS_EXPOSED": {
                "prompt": "wearing black athletic leggings, athletic wear",
                "priority": 2
            }
        }

        self.confidence_threshold = 0.55
        self.seed = 42
        self.clothing_template = None
        self.initial_prompt = None
        self.processing_params = {
            'guidance_scale': 9.0,
            'num_inference_steps': 50,
            'strength': 0.99
        }

        self.style_modifiers = (
            ", athletic photoshoot, professional sports photography, studio lighting, "
            "8k uhd, detailed black fabric texture, perfect fit athletic wear, "
            "professional sports clothing, sharp focus, high contrast"
        )
        
        self.negative_prompt = (
            "nude, naked, revealing, inappropriate, low quality, blurry, distorted, "
            "deformed, disfigured, bad anatomy, wrinkled clothing, changing colors, "
            "color variation, pattern variation, design variation, inconsistent style, "
            "bad proportions, duplicate, morbid, mutilated, poorly drawn, non-athletic wear"
        )

    def create_enhanced_mask(self, image, detections):
        mask = Image.new('RGB', image.size, 'black')
        draw = ImageDraw.Draw(mask)

        sorted_detections = sorted(
            detections,
            key=lambda x: self.target_classes[x['class']]['priority']
        )

        for detection in sorted_detections:
            if detection['score'] >= self.confidence_threshold:
                box = detection['box']
                x0, y0, w, h = box
                x1, y1 = x0 + w, y0 + h
                
                padding = int(min(w, h) * 0.15)
                x0 = max(0, x0 - padding)
                y0 = max(0, y0 - padding)
                x1 = min(image.size[0], x1 + padding)
                y1 = min(image.size[1], y1 + padding)
                
                draw.rectangle([x0, y0, x1, y1], fill='white')

        return mask

    def get_optimized_prompt(self, detections):
        prompt_parts = set()
        
        for detection in detections:
            if detection['score'] >= self.confidence_threshold:
                prompt_parts.add(self.target_classes[detection['class']]['prompt'])

        base_prompt = "professional sports photograph of a person " + ", ".join(prompt_parts)
        return f"{base_prompt}{self.style_modifiers}"

    def process_frame(self, frame, is_first_frame=False):
        try:
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_frame_path = temp_file.name
                frame_pil.save(temp_frame_path, quality=95)

            try:
                detections = self.nude_detector.detect(temp_frame_path)
                
                filtered_detections = [
                    det for det in detections
                    if det['class'] in self.target_classes and det['score'] >= self.confidence_threshold
                ]

                if not filtered_detections:
                    return frame

                mask = self.create_enhanced_mask(frame_pil, filtered_detections)
                
                if is_first_frame or not self.initial_prompt:
                    self.initial_prompt = self.get_optimized_prompt(filtered_detections)

                output = self.pipe(
                    prompt=self.initial_prompt,
                    negative_prompt=self.negative_prompt,
                    image=frame_pil,
                    mask_image=mask,
                    num_inference_steps=self.processing_params['num_inference_steps'],
                    guidance_scale=self.processing_params['guidance_scale'],
                    strength=self.processing_params['strength'],
                    seed=self.seed
                ).images[0]

                if is_first_frame:
                    self.clothing_template = output

                output = output.resize(frame_pil.size, Image.Resampling.LANCZOS)
                output_array = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)

                return output_array

            except Exception as e:
                self.logger.error(f"Error processing frame: {str(e)}")
                return frame

            finally:
                if os.path.exists(temp_frame_path):
                    os.unlink(temp_frame_path)

        except Exception as e:
            self.logger.error(f"Critical error in frame processing: {str(e)}")
            return frame

    def process_video(self, input_path, output_path, frame_skip=1, start_time=0, duration=None):
        """Process video with enhanced error handling and frame consistency"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")

        self.logger.info(f"Starting video processing: {input_path}")
        cap = cv2.VideoCapture(input_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame range
        start_frame = int(start_time * fps)
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if duration is None else start_frame + int(duration * fps)
        
        # Adjust output FPS based on frame skip
        output_fps = fps // frame_skip

        # Create temporary output file
        temp_output = f"{os.path.splitext(output_path)[0]}_temp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, output_fps, (width, height))

        if not out.isOpened():
            raise RuntimeError("Failed to create video writer")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        try:
            with tqdm(total=(end_frame - start_frame) // frame_skip) as pbar:
                frame_count = start_frame
                first_frame = True
                frame_buffer = []  # Buffer for frame smoothing

                while cap.isOpened() and frame_count < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % frame_skip == 0:
                        processed_frame = self.process_frame(frame, is_first_frame=first_frame)
                        
                        # Frame smoothing logic
                        if first_frame:
                            first_frame = False
                            # Initialize buffer with first processed frame
                            frame_buffer = [processed_frame] * 3
                        else:
                            # Update buffer
                            frame_buffer.pop(0)
                            frame_buffer.append(processed_frame)
                            # Average the frames for smoothing
                            processed_frame = np.mean(frame_buffer, axis=0).astype(np.uint8)

                        if processed_frame is not None and processed_frame.shape == (height, width, 3):
                            out.write(processed_frame)
                        else:
                            self.logger.warning(f"Invalid frame at position {frame_count}")
                            out.write(frame)
                        
                        pbar.update(1)

                    frame_count += 1

        except Exception as e:
            self.logger.error(f"Error during video processing: {str(e)}")
            raise

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

            if os.path.exists(temp_output):
                self.logger.info("Converting to final output format...")
                # Use better quality settings for final encoding
                os.system(f'ffmpeg -i {temp_output} -c:v libx264 -preset slow -crf 18 -c:a aac "{output_path}"')
                os.remove(temp_output)
                self.logger.info("Video processing completed successfully")

def main():
    try:
        generator = EnhancedVideoClothingGenerator()
        
        input_video = "/kaggle/input/in-painting-sample/In_Painting_sample.mp4"  # Replace with your input video path
        output_video = "processed_output.mp4"      # Replace with your desired output path
        
        generator.process_video(
            input_path=input_video,
            output_path=output_video,
            frame_skip=1,  # Process every frame for better consistency
            start_time=0,
            duration=None
        )

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()