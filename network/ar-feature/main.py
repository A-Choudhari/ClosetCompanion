import cv2
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import torch
import argparse

from models.sams_model import SamsModel  # Adjust based on actual path
from pytorch_lightning import Trainer

def load_online_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def main():
    # Example URL of the clothing image
    image_url = "data:image/webp;base64,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"
    cloth_image = load_online_image(image_url)

    # Define hyperparameters
    hparams = {
        'n_frames_total': 5,
        'person_inputs': ('agnostic', 'densepose', 'flow'),
        'cloth_inputs': ('cloth',),
        'flow_warp': True,
        'init_type': 'normal',
        'init_variance': 0.02,
        'gan_mode': 'hinge',
        'lr': 0.0002,
        'lr_D': 0.0002,
        'display_count': 100
    }

    # Convert hparams to argparse.Namespace
    hparams = argparse.Namespace(**hparams)

    # Initialize the SamsModel
    tryon_model = SamsModel(hparams)

    # Load pre-trained weights if available
    # tryon_model.load_state_dict(torch.load('path_to_weights.pth'))

    # Set model to evaluation mode
    tryon_model.eval()

    # OpenCV live capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to Tensor
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        cloth_tensor = torch.from_numpy(cloth_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Apply the virtual try-on
        with torch.no_grad():
            tryon_result = tryon_model.generator(frame_tensor, cloth_tensor)  # Adjust if needed

        # Convert Tensor back to image
        tryon_result = tryon_result.squeeze().permute(1, 2, 0).numpy() * 255
        tryon_result = tryon_result.astype(np.uint8)

        # Display the result
        cv2.imshow("Virtual Try-On", tryon_result)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
