import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import os
from torchreid.models import build_model
from pinecone import Pinecone
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image

def main():
    load_dotenv()
    
    # Initialize Pinecone API
    API_KEY = os.getenv('PINECONE_API_KEY')
    pc = Pinecone(api_key=API_KEY)
    
    IMAGE_FOLDER_PATH = 'data_test'

    # Connect to existing index
    index = pc.Index("image-index")
    
    # Load pretrained model
    model = build_model(name='osnet_x1_0', num_classes=751, pretrained=True)
    model.eval()

    # Enable CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Preprocess function
    def preprocess_image(image, target_size):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0).to(device)

    # Function to extract features from an image
    def extract_features(image, target_size):
        processed_image = preprocess_image(image, target_size)
        with torch.no_grad():
            feature = model(processed_image).squeeze().cpu().numpy()
        return feature

    # Process and upload images from folder
    def upload_images_to_pinecone(IMAGE_FOLDER_PATH):
        target_size = (256, 128)  # Model input size for OSNet
        for file_name in os.listdir(IMAGE_FOLDER_PATH):
            if file_name.lower().endswith(('jpg', 'png', 'jpeg')):
                image_path = os.path.join(IMAGE_FOLDER_PATH, file_name)
                image = cv2.imread(image_path)
                if image is not None:
                    feature_vector = extract_features(image, target_size)
                    metadata = {'file_name': file_name}
                    index.upsert(
                        [
                            (file_name, feature_vector.tolist(), metadata)
                        ]
                    )
                    print(f'Uploaded {file_name} to Pinecone.')

    upload_images_to_pinecone(IMAGE_FOLDER_PATH)

if __name__ == '__main__':
    main()