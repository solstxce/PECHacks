import torch
from PIL import Image, ImageDraw
from nudenet import NudeDetector
from diffusers import StableDiffusionInpaintPipeline
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os


class ClothingGenerator:
    def __init__(self):
        # Initialize NudeNet detector
        self.nude_detector = NudeDetector()

        # Initialize Stable Diffusion pipeline
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")

        # Classes we want to detect and replace
        self.target_classes = [
            "ANUS_EXPOSED",
            "BUTTOCKS_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
            "FEMALE_GENITALIA_COVERED",
            "FEMALE_BREAST_COVERED",
            "FEMALE_GENITALIA_EXPOSED"
        ]

        # Confidence threshold
        self.confidence_threshold = 0.50

        # Mapping of classes to clothing prompts
        self.clothing_prompts = {
            "FEMALE_BREAST_EXPOSED": "wearing a casual T-shirt",
            "FEMALE_BREAST_COVERED": "wearing a casual T-shirt",
            "BUTTOCKS_EXPOSED": "wearing casual jeans",
            "FEMALE_GENITALIA_EXPOSED": "wearing full-length jeans",
            "FEMALE_GENITALIA_COVERED": "wearing full-length jeans",
            "MALE_GENITALIA_EXPOSED": "wearing casual jeans",
            "ANUS_EXPOSED": "wearing casual jeans"
        }

    def create_mask_from_detections(self, image, detections):
        """Create a mask image from detection boxes"""
        mask = Image.new('RGB', image.size, 'black')
        draw = ImageDraw.Draw(mask)

        for detection in detections:
            if (detection['class'] in self.target_classes and
                detection['score'] >= self.confidence_threshold):
                # Get box coordinates
                x0, y0, w, h = detection['box']
                x1, y1 = x0 + w, y0 + h

                # Draw white rectangle on mask for area to inpaint
                draw.rectangle([x0, y0, x1, y1], fill='white')

        return mask

    def expand_box(self, box, image_size, margin=20):
        """Expand detection box by margin pixels"""
        x0, y0, w, h = box
        x1, y1 = x0 + w, y0 + h

        # Expand box
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(image_size[0], x1 + margin)
        y1 = min(image_size[1], y1 + margin)

        return [x0, y0, x1 - x0, y1 - y0]

    def process_image(self, image_path):
        """Process single image - detect and replace inappropriate content"""
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Get detections
        detections = self.nude_detector.detect(image_path)

        # Filter and process detections
        filtered_detections = []
        for detection in detections:
            if (detection['class'] in self.target_classes and
                detection['score'] >= self.confidence_threshold):
                # Expand box slightly to ensure better coverage
                detection['box'] = self.expand_box(detection['box'], image.size)
                filtered_detections.append(detection)
                print(f"Detected {detection['class']} with confidence {detection['score']:.2f}")

        if not filtered_detections:
            print("No inappropriate content detected.")
            return image

        # Create mask for inpainting
        mask = self.create_mask_from_detections(image, filtered_detections)

        # Generate appropriate prompt based on detections
        prompt_parts = []
        for detection in filtered_detections:
            prompt_parts.append(self.clothing_prompts[detection['class']])
        base_prompt = "professional photograph of a person " + ", ".join(set(prompt_parts))
        prompt = f"{base_prompt}, high quality, detailed, natural lighting"

        # Run inpainting
        output = self.pipe(
            prompt=prompt,
            negative_prompt="nude, naked, revealing, inappropriate, low quality, blurry",
            image=image,
            mask_image=mask,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]

        return output

    def process_directory(self, input_pattern, output_dir="output"):
        """Process multiple images matching the input pattern"""
        os.makedirs(output_dir, exist_ok=True)

        image_paths = glob(input_pattern)
        print(f"Found {len(image_paths)} images to process")

        for i, image_path in enumerate(image_paths):
            print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path}")
            try:
                output_image = self.process_image(image_path)

                # Save output
                output_path = os.path.join(output_dir, f"processed_{os.path.basename(image_path)}")
                output_image.save(output_path)
                print(f"Saved processed image to {output_path}")

                # Display before/after
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
                #ax1.imshow(Image.open(image_path))
                ax1.set_title("Original")
                ax1.axis('off')
                ax2.imshow(output_image)
                ax2.set_title("Processed")
                ax2.axis('off')
                plt.show()

            except Exception as e:
                print(f"Error processing {image_path}: {e}")


# Example usage
def main():
    # Initialize the generator
    generator = ClothingGenerator()

    # Process single image
    image_path = "/content/samantha_202461.jpg"  # Replace with your image path
    output = generator.process_image(image_path)

    # Display result
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(Image.open(image_path))
    plt.title("Original")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.title("Processed")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
