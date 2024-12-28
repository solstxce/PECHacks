import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file, render_template
import base64
import io
import os
from flask_cors import CORS

# Define the necessary components of the Xception network
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        return self.pointwise(x)

class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super().__init__()
        self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False) if (out_filters != in_filters or strides != 1) else None
        self.skipbn = nn.BatchNorm2d(out_filters) if self.skip is not None else None

        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters

        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for _ in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        skip = self.skip(x) if self.skip is not None else x
        if self.skipbn is not None:
            skip = self.skipbn(skip)
        return self.rep(x) + skip

class Xception(nn.Module):
    def __init__(self, num_classes=2, inc=3, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.last_linear = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = torch.mean(x, dim=[2, 3])  # Global average pooling
        x = self.last_linear(x)
        return x

# Define the XceptionDetector class
class XceptionDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = Xception(
            num_classes=config['backbone_config']['num_classes'],
            inc=config['backbone_config']['inc'],
            dropout=config['backbone_config']['dropout']
        )
        state_dict = torch.load(config['pretrained'], map_location=torch.device('cpu'))
        self.backbone.load_state_dict(state_dict, strict=False)

    def forward(self, data_dict):
        return self.backbone(data_dict['image'])

# Initialize Flask app
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Define the transformations globally
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize the model globally
config = {
    'backbone_config': {
        'num_classes': 2,
        'inc': 3,
        'dropout': 0.5,
    },
    'pretrained': 'xception_best.pth',
}

model = XceptionDetector(config)
state_dict = torch.load(config['pretrained'], map_location=torch.device('cpu'), weights_only=True)
model.backbone.load_state_dict(state_dict, strict=False)
model.eval()

# Add a route for the root URL
@app.route('/')
def home():
    return send_file('index.html')

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    try:
        # Get the image from the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        
        # Process the image
        image_tensor = transform(image).unsqueeze(0)
        
        # Perform inference
        with torch.no_grad():
            outputs = model({'image': image_tensor})
            # Get predictions
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities).item()
            confidence = probabilities.max().item()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Explicitly clear variables
        del image_tensor
        del outputs
        del probabilities
        
        # Prepare response
        result = {
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': float(confidence),
            'status': 'success'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500
    finally:
        # Ensure garbage collection runs
        import gc
        gc.collect()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
