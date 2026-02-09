import torch
import torch.nn.functional as F
import cv2
import numpy as np

def post_process_image(
    img_tensor,
    apply_sharpen='light'   # heavy , light , None
):
    """
    Light post-processing for pix2pix outputs.
    Visualization ONLY.
    """

    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]

    # [-1, 1] → [0, 255]
    img = (img_tensor.clamp(-1, 1) + 1) / 2
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)

    if apply_sharpen == 'light':
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.addWeighted(img, 1.1, blurred, -0.1, 0)
    
    if apply_sharpen == "heavy":
        blurred = cv2.GaussianBlur(img, (5, 5), 0)        
        # Heavier sharpening 
        sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    return img

def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)


# postprocessing/post_process.py
if __name__ == "__main__":
    import sys
    import torch
    import cv2

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    tensor = torch.load(input_path)
    img = post_process_image(tensor, apply_sharpen='heavy')

    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
