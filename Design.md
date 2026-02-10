# Satellite-to-Map Image Translation — Design Document

## 1. Project Overview

This project implements **Conditional Generative Adversarial Networks (cGANs)** to translate **satellite images** into **map-style images**. The system uses a **pretrained pix2pix generator** fine-tuned on 128x128 images and is designed to be modular, reproducible, and Colab GPU-friendly.

---

## 2. Input and Output

**Input:**

* Satellite image, 128×128 pixels, RGB
* Normalized to [-1, 1]

**Output:**

* Map image, 128x128 pixels, RGB
* Normalized to [-1, 1]

**Train/Validation Split:** 70% train / 15% validation / 15% test

---

## 3. Generator Architecture (U-Net)

* **Type:** U-Net encoder-decoder with skip connections
* **Input:** 128×128×3
* **Encoder:** 5 layers, filters increasing from base_filters 64
* **Decoder:** 5 layers, mirrors encoder, skip connections applied
* **Output Activation:** tanh
* **Weights:** Training from a scrach with a pix2pix dataset

---

## 4. Discriminator Architecture (PatchGAN)

* **Type:** PatchGAN
* **Input:** Generated or real map image concatenated with corresponding satellite image
* **Patch Size:** 32×32 (for 128×128 images)
* **Layers:** Convolutional layers with LeakyReLU activations
* **Output:** Patch-level real/fake scores

---

## 5. Loss Functions and Weights

* **L1 Reconstruction Loss:** λ_L1 = 200
* **Adversarial Loss (GAN):** encourages realism
* **SSIM Loss:** improves structural similarity

---

## 6. Metrics for Evaluation

* **PSNR (Peak Signal-to-Noise Ratio)**
* **SSIM (Structural Similarity Index)**
* **Qualitative Visualization:** side-by-side comparison:

  * Satellite image
  * Generated map image
  * Ground truth map image

---

## 7. Training Strategy (cpu)

* **Batch Size:** 8
* **Optimizer:** Adam (lr = 2e-4, β1 = 0.5)
* **Epochs:** 50 (max) for 128×128
* **Data Augmentation:** random flips, rotations
* **Training Phases:**

  1. **Baseline:** generator only, L1 loss, sanity check and benchmark
  2. **Full pix2pix GAN:** generator + discriminator, L1 + adversarial loss + ssim loss

---

## 8. System Diagram

```
Satellite Image (128x128) 
         |
         v
   [Generator U-Net]
         |
Generated Map (128x128)
         |
         v
   [PatchGAN Discriminator]
         |
Real / Fake Score
```

