import torch
from PIL import Image, ImageDraw
import cv2
import numpy as np
from nudenet import NudeDetector
from diffusers import StableDiffusionInpaintPipeline
import os
from tqdm import tqdm
import tempfile

class VideoClothingGenerator:
    def __init__(self):
        self.nude_detector = NudeDetector()

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")

        self.target_classes = [
            "ANUS_EXPOSED",
            "BUTTOCKS_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
            "FEMALE_GENITALIA_COVERED",
            "FEMALE_BREAST_COVERED",
            "FEMALE_GENITALIA_EXPOSED"
        ]

        self.confidence_threshold = 0.50
        self.clothing_template = None
        self.initial_prompt = None

        self.clothing_prompts = {
            "FEMALE_BREAST_EXPOSED": "wearing a blue T-shirt",
            "FEMALE_BREAST_COVERED": "wearing a blue T-shirt",
            "BUTTOCKS_EXPOSED": "wearing blue jeans",
            "FEMALE_GENITALIA_EXPOSED": "wearing full-length blue jeans",
            "FEMALE_GENITALIA_COVERED": "wearing full-length blue jeans",
            "MALE_GENITALIA_EXPOSED": "wearing blue jeans",
            "ANUS_EXPOSED": "wearing blue jeans"
        }

    def create_mask_from_detections(self, image, detections):
        mask = Image.new('RGB', image.size, 'black')
        draw = ImageDraw.Draw(mask)

        for detection in detections:
            if (detection['class'] in self.target_classes and
                detection['score'] >= self.confidence_threshold):
                x0, y0, w, h = detection['box']
                x1, y1 = x0 + w, y0 + h
                draw.rectangle([x0, y0, x1, y1], fill='white')

        return mask

    def expand_box(self, box, image_size, margin=20):
        x0, y0, w, h = box
        x1, y1 = x0 + w, y0 + h

        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(image_size[0], x1 + margin)
        y1 = min(image_size[1], y1 + margin)

        return [x0, y0, x1 - x0, y1 - y0]

    def process_first_frame(self, frame):
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_frame_path = temp_file.name
            frame_pil.save(temp_frame_path, quality=95)

        try:
            detections = self.nude_detector.detect(temp_frame_path)

            filtered_detections = []
            for detection in detections:
                if (detection['class'] in self.target_classes and
                    detection['score'] >= self.confidence_threshold):
                    detection['box'] = self.expand_box(detection['box'], frame_pil.size)
                    filtered_detections.append(detection)

            if not filtered_detections:
                self.initial_prompt = "professional photograph of a person wearing general clothing"
                print("Warning: No detections found, using default fallback prompt.")
                return frame, None, None

            mask = self.create_mask_from_detections(frame_pil, filtered_detections)

            prompt_parts = []
            for detection in filtered_detections:
                prompt_parts.append(self.clothing_prompts[detection['class']])
            base_prompt = "professional photograph of a person " + ", ".join(set(prompt_parts))
            self.initial_prompt = f"{base_prompt}, high quality, detailed, natural lighting"

            output = self.pipe(
                prompt=self.initial_prompt,
                negative_prompt="nude, naked, revealing, inappropriate, low quality, blurry",
                image=frame_pil,
                mask_image=mask,
                num_inference_steps=50,
                guidance_scale=7.5,
                seed=42
            ).images[0]

            self.clothing_template = output
            output = output.resize(frame_pil.size, Image.Resampling.LANCZOS)
            output_array = np.array(output, dtype=np.uint8)
            return cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR), filtered_detections, mask

        finally:
            os.unlink(temp_frame_path)

    def process_subsequent_frame(self, frame, reference_detections):
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not self.initial_prompt:
            self.initial_prompt = "professional photograph of a person wearing general clothing"

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_frame_path = temp_file.name
            frame_pil.save(temp_frame_path, quality=95)

        try:
            detections = self.nude_detector.detect(temp_frame_path)

            filtered_detections = []
            for detection in detections:
                if (detection['class'] in self.target_classes and
                    detection['score'] >= self.confidence_threshold):
                    detection['box'] = self.expand_box(detection['box'], frame_pil.size)
                    filtered_detections.append(detection)

            if not filtered_detections:
                return frame

            mask = self.create_mask_from_detections(frame_pil, filtered_detections)

            output = self.pipe(
                prompt=self.initial_prompt,
                negative_prompt="nude, naked, revealing, inappropriate, low quality, blurry",
                image=frame_pil,
                mask_image=mask,
                num_inference_steps=30,
                guidance_scale=7.5,
                seed=42
            ).images[0]

            output = output.resize(frame_pil.size, Image.Resampling.LANCZOS)
            output_array = np.array(output, dtype=np.uint8)
            return cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)

        finally:
            os.unlink(temp_frame_path)

    def process_video(self, input_path, output_path, frame_skip=1, start_time=0, duration=None):
        """Process video with frame skipping"""
        cap = cv2.VideoCapture(input_path)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_time * fps)
        if duration is not None:
            end_frame = start_frame + int(duration * fps)
        else:
            end_frame = total_frames

        # Adjust output FPS based on frame skip
        output_fps = fps // frame_skip

        temp_output = f"{os.path.splitext(output_path)[0]}_temp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, output_fps, (width, height))

        if not out.isOpened():
            raise Exception("Failed to create video writer")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        try:
            with tqdm(total=(end_frame - start_frame) // frame_skip) as pbar:
                frame_count = start_frame
                first_frame = True
                reference_detections = None

                while cap.isOpened() and frame_count < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if first_frame:
                        processed_frame, reference_detections, _ = self.process_first_frame(frame)
                        first_frame = False
                    else:
                        if frame_count % frame_skip == 0:
                            processed_frame = self.process_subsequent_frame(frame, reference_detections)
                        else:
                            frame_count += 1
                            continue

                    if processed_frame is not None and processed_frame.shape == (height, width, 3):
                        out.write(processed_frame)
                    else:
                        print(f"Warning: Invalid frame at position {frame_count}")
                        out.write(frame)

                    frame_count += 1
                    if frame_count % frame_skip == 0:
                        pbar.update(1)

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

            if os.path.exists(temp_output):
                os.system(f'ffmpeg -i {temp_output} -c:v libx264 -preset medium -crf 23 {output_path}')
                os.remove(temp_output)

def main():
    generator = VideoClothingGenerator()

    input_video = "/content/billa_movie.mp4"
    output_video = "output_vide1o.mp4"

    # Process video with frame skipping (e.g., process every 5th frame)
    frame_skip = 1
    generator.process_video(input_video, output_video, frame_skip=frame_skip)

if __name__ == "__main__":
    main()
