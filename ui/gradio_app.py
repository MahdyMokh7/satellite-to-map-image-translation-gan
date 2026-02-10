import torch
import gradio as gr
from PIL import Image
import torchvision.transforms as T
import numpy as np
import os
import sys
import yaml

# --- Project setup ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from models.generator import GeneratorUNet
from preprocessing.preprocess_data import preprocess_image_for_ui
from postprocessing.post_process import post_process_image


device = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = os.path.join(root_dir, os.path.join("results", "checkpoints", "cgan", "generator_best.pth"))

config_file_path = os.path.join(root_dir, 'configs', 'config.yaml')
with open(config_file_path, "r") as f:
    config = yaml.safe_load(f)

# ---- Load model ----
model = GeneratorUNet(
    in_channels=config['generator']['in_channels'], 
    out_channels=config['generator']['out_channels'], 
    base_filters=config['generator']['base_filters'], 
).to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


preprocess_image = preprocess_image_for_ui
postprocess_image = post_process_image


def generate_map(input_image: Image.Image):
    """
    Generates a map from an input satellite image using the model.
    
    Args:
        input_image (PIL.Image): The input image (satellite image).
    Returns:
        PIL.Image: The generated map image.
    """
    input_tensor = preprocess_image(input_image, device)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_image = postprocess_image(output_tensor, apply_sharpen='heavy')
    
    return output_image


interface = gr.Interface(
    fn=generate_map,
    inputs=gr.Image(type="pil", label="Satellite Image"),
    outputs=gr.Image(type="pil", label="Generated Map"),
    title="Satellite → Map Translation (pix2pix GAN)",
    description="Upload a satellite image and generate the corresponding map using a trained conditional GAN."
)


if __name__ == "__main__":
    interface.launch()
