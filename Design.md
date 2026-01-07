# Satellite-to-Map Image Translation — Design Document

## 1. Project Overview

This project implements **Conditional Generative Adversarial Networks (cGANs)** to translate **satellite images** into **map-style images**. The system uses a **pretrained pix2pix generator** fine-tuned on 64×64 images and is designed to be modular, reproducible, and Colab GPU-friendly.

---

## 2. Input and Output

**Input:**

* Satellite image, 64×64 pixels, RGB
* Normalized to [-1, 1]

**Output:**

* Map image, 64×64 pixels, RGB
* Normalized to [-1, 1]

**Train/Validation Split:** 80% train / 20% validation

---

## 3. Generator Architecture (U-Net, Pretrained)

* **Type:** U-Net encoder-decoder with skip connections
* **Input:** 64×64×3
* **Encoder:** 5 layers, filters increasing 32 → 256
* **Decoder:** 5 layers, mirrors encoder, skip connections applied
* **Output Activation:** tanh
* **Pretrained Weights:** Fine-tuning from a similar dataset
* **Fine-tuning Strategy:**

  * Freeze first 2–3 layers initially
  * Unfreeze all layers after stabilization

---

## 4. Discriminator Architecture (PatchGAN)

* **Type:** PatchGAN
* **Input:** Generated or real map image concatenated with corresponding satellite image
* **Patch Size:** 16×16 (for 64×64 images)
* **Layers:** Convolutional layers with LeakyReLU activations
* **Output:** Patch-level real/fake scores

---

## 5. Loss Functions and Weights

* **L1 Reconstruction Loss:** λ_L1 = 100
* **Adversarial Loss (GAN):** encourages realism
* **Optional SSIM Loss:** improves structural similarity

---

## 6. Metrics for Evaluation

* **PSNR (Peak Signal-to-Noise Ratio)**
* **SSIM (Structural Similarity Index)**
* **Qualitative Visualization:** side-by-side comparison:

  * Satellite image
  * Generated map image
  * Ground truth map image

---

## 7. Training Strategy (Colab GPU Constraints)

* **Batch Size:** 8
* **Optimizer:** Adam (lr = 2e-4, β1 = 0.5)
* **Epochs:** 50–100 for 64×64 prototyping
* **Data Augmentation:** random flips, rotations
* **Training Phases:**

  1. **Baseline:** generator only, L1 loss, sanity check and benchmark
  2. **Full pix2pix GAN:** generator + discriminator, L1 + adversarial loss

---

## 8. System Diagram

```
Satellite Image (64x64) 
         |
         v
   [Generator U-Net]
         |
Generated Map (64x64)
         |
         v
   [PatchGAN Discriminator]
         |
Real / Fake Score
```

