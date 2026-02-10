###########################################
# Model Load Part
###########################################

import torch
from PIL import Image
import yaml
import sys
import os

# --- Project setup ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)

from models.generator import GeneratorUNet
from preprocessing.preprocess_data import preprocess_image_for_ui
from postprocessing.post_process import post_process_image

device = "cuda" if torch.cuda.is_available() else "cpu"
configs_path = os.path.join(root_dir, "configs", "config.yaml")

# Load config
with open(configs_path) as f:
    config = yaml.safe_load(f)

checkpoint_path = os.path.join(root_dir, config['outputs']['checkpoints_dir'], 'generator_best.pth')

model = GeneratorUNet(
    in_channels=config["generator"]["in_channels"],
    out_channels=config["generator"]["out_channels"],
    base_filters=config["generator"]["base_filters"],
).to(device)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()




##################################################
# Flask Part
##################################################

import os
import sys
import numpy as np
from flask import Flask, render_template, request

UPLOAD_FOLDER = os.path.join(root_dir, "ui", "flask_app", "static", "uploads")
OUTPUTS_FOLDER = os.path.join(root_dir, "ui", "flask_app", "static", "outputs")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUTS_FOLDER"] = OUTPUTS_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    image_file = request.files["image"]

    # Save uploaded file
    input_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(input_path)

    # Generate unique output path
    import uuid
    output_filename = f"generated_{uuid.uuid4().hex}.png"
    output_path = os.path.join(OUTPUTS_FOLDER, output_filename)

    # Open image and preprocess
    input_image = Image.open(input_path).convert("RGB")
    input_tensor = preprocess_image_for_ui(input_image, device)

    # Model inference
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Post-processing
    output_image = post_process_image(output_tensor, apply_sharpen='heavy')

    # Convert to PIL Image if still a NumPy array
    if isinstance(output_image, np.ndarray):
        output_image = Image.fromarray(output_image.astype('uint8'))

    # Save output
    output_image.save(output_path)

    # Render template with input/output
    return render_template(
        "index.html",
        input_image=f"uploads/{image_file.filename}",  # relative to static
        output_image=f"outputs/{output_filename}"     # relative to static
    )



if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

    app.run(debug=True)
