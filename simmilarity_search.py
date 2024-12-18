import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import os
import json
from torchreid.models import build_model
from torch.nn.functional import cosine_similarity
from pinecone import Pinecone
from dotenv import load_dotenv


def save_data_to_nosql(data, folder_name='data_nosql'):
    """Save reference data locally in a streamlined format."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_path = os.path.join(folder_name, 'reference_data.json')
    with open(file_path, 'w') as f:
        json.dump(data, f)


def load_data_from_nosql(folder_name='data_nosql'):
    """Load reference data from the local JSON file."""
    file_path = os.path.join(folder_name, 'reference_data.json')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}


def compare_vectors(input_vector, reference_data, top_k=5, min_score=7.0):
    """Compare input feature vector with reference vectors."""
    similarities = []
    for filename, ref_vector in reference_data.items():
        ref_vector = np.array(ref_vector)
        score = cosine_similarity(
            torch.tensor(input_vector).unsqueeze(0),
            torch.tensor(ref_vector).unsqueeze(0)
        ).item()
        if score >= min_score:
            similarities.append((filename, score))
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]


def main():
    """Main function to perform image similarity search."""
    load_dotenv()
    API_KEY = os.getenv('PINECONE_API_KEY')
    pc = Pinecone(api_key=API_KEY)
    index_name = "image-index"
    index = pc.Index(index_name)

    # Load pre-trained model
    model = build_model(name='osnet_x1_0', num_classes=751, pretrained=False)
    checkpoint = torch.load('model/osnet_x1_0.tar-60', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def preprocess_image(image, target_size):
        """Preprocess the image for feature extraction."""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0).to(device)

    def extract_features(image, target_size):
        """Extract feature vectors from the input image."""
        processed_image = preprocess_image(image, target_size)
        with torch.no_grad():
            feature = model(processed_image).squeeze().cpu().numpy()
        return feature

    # Fetch reference data from Pinecone
    total_vectors = index.describe_index_stats().get('total_vector_count', 0)
    response = index.fetch(ids=list(range(total_vectors)))
    reference_data = {}
    for item in response['vectors'].values():
        filename = item['metadata']['file_name']
        feature = item['values']
        reference_data[filename] = feature

    # Save reference data locally
    save_data_to_nosql(reference_data)

    # Initialize webcam capture
    cap = cv2.VideoCapture(0)
    target_size = (256, 128)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract features from webcam frame
        frame_features = extract_features(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), target_size)

        # Load reference data from local storage
        reference_data = load_data_from_nosql()

        # Perform similarity search
        matches = compare_vectors(frame_features, reference_data)

        # Display results
        result_text = 'No confident match found'
        if matches:
            best_match, best_score = matches[0]
            result_text = f'Best Match: {best_match} ({best_score:.2f})'

        # Overlay result text on webcam frame
        cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame in a window
        cv2.imshow('Image Similarity Search', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
