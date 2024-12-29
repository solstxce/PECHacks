from flask import Flask, request, send_file
from pyngrok import ngrok, conf
from flask_cors import CORS
import torch
from PIL import Image, ImageDraw
from nudenet import NudeDetector
from diffusers import StableDiffusionInpaintPipeline
import io
import os
from werkzeug.utils import secure_filename

# Ngrok setup
ngrok_auth_token = "2qsD7bDOIt7Bx7qDWaelfdOirk9_6AV7HnRCyppwV7FcHqyDn"
conf.get_default().auth_token = ngrok_auth_token

class ClothingGenerator:
    def __init__(self):  # Fixed initialization method name
        # Initialize NudeNet detector
        self.nude_detector = NudeDetector()

        # Initialize Stable Diffusion pipeline
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
        
        self.clothing_prompts = {
            "FEMALE_BREAST_EXPOSED": "wearing a long modest full-length dress with high neckline",
            "FEMALE_BREAST_COVERED": "wearing a conservative full-length dress",
            "BUTTOCKS_EXPOSED": "wearing a long flowing skirt",
            "FEMALE_GENITALIA_EXPOSED": "wearing a long modest dress",
            "FEMALE_GENITALIA_COVERED": "wearing a full-length conservative dress",
            "MALE_GENITALIA_EXPOSED": "wearing loose fitting pants and long shirt",
            "ANUS_EXPOSED": "wearing a long loose dress"
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

    def process_image(self, image_path):
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Get detections
        detections = self.nude_detector.detect(image_path)

        # Filter and process detections
        filtered_detections = []
        for detection in detections:
            if (detection['class'] in self.target_classes and
                detection['score'] >= self.confidence_threshold):
                detection['box'] = self.expand_box(detection['box'], image.size)
                filtered_detections.append(detection)

        if not filtered_detections:
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

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = '/kaggle/working/temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize the generator
generator = ClothingGenerator()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return '''
    <html>
        <body>
            <h2>Upload Image for Processing</h2>
            <form action="/process_image" method="post" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*">
                <input type="submit" value="Process Image">
            </form>
        </body>
    </html>
    '''

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return 'No image file provided', 400

        file = request.files['image']
        
        if file.filename == '':
            return 'No selected file', 400

        if not allowed_file(file.filename):
            return 'File type not allowed', 400

        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        
        with open(temp_path, 'wb') as f:
            file.save(temp_path)

        try:
            processed_image = generator.process_image(temp_path)
            img_byte_arr = io.BytesIO()
            processed_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            return send_file(
                img_byte_arr,
                mimetype='image/png',
                as_attachment=True,
                download_name='processed_image.png'
            )

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        return str(e), 500

def main():
    # Start ngrok
    http_tunnel = ngrok.connect(5000)
    print(f' * Tunnel URL: {http_tunnel.public_url}')
    
    # Run app without debug mode
    app.run(port=5000, host='0.0.0.0')

if __name__ == '__main__':
    main()
