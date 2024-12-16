import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import os
from torchreid.models import build_model
from torch.nn.functional import cosine_similarity

def main():
    # Load pretrained model
    model = build_model(name='osnet_x1_0', num_classes=751, pretrained=False)
    checkpoint = torch.load('model\osnet_x1_0.tar-60', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['state_dict'])
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

    # Load reference images
    def load_reference_images(reference_folder, target_size):
        reference_features = {}
        for filename in os.listdir(reference_folder):
            filepath = os.path.join(reference_folder, filename)
            image = cv2.imread(filepath)
            if image is not None:
                processed_image = preprocess_image(image, target_size)
                with torch.no_grad():
                    feature = model(processed_image).squeeze()
                reference_features[filename] = feature
        return reference_features

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    reference_folder = 'data_test'
    target_size = (256, 128)  # Model input size for OSNet
    reference_features = load_reference_images(reference_folder, target_size)

    MIN_CONFIDENCE = 0.7

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), target_size)
        with torch.no_grad():
            query_feature = model(processed_frame).squeeze()

        # Compare with reference images
        best_match = "No match"
        highest_similarity = -float('inf')

        for filename, feature in reference_features.items():
            similarity = cosine_similarity(query_feature, feature, dim=0).item()
            if similarity > highest_similarity:
                best_match = filename
                highest_similarity = similarity

        # Display result if above minimum confidence
        if highest_similarity >= MIN_CONFIDENCE:
            result_text = f'Best Match: {best_match} ({highest_similarity:.2f})'
        else:
            result_text = 'No confident match found'

        cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Image Similarity Search', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()