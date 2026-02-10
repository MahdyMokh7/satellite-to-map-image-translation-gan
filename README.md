# Satellite-to-Map Image Translation using Conditional GANs (pix2pix)

## 📌 Project Overview

This project focuses on **image-to-image translation** using **Conditional Generative Adversarial Networks (cGANs)** to convert **satellite imagery** into **map-style (Google Maps–like) representations**. The goal is to learn a robust pixel-to-pixel mapping that preserves spatial structure while translating visual semantics from one domain to another.

The project is designed to be **industry-ready**, reproducible, and aligned with real-world computer vision workflows used in **remote sensing, geospatial intelligence, and mapping systems**.

---

## 🎯 Problem Definition

Given a satellite image as input, the model generates a corresponding map-style image:

**Satellite Image → Map Image**

Key challenges addressed:

* Preserving spatial and structural consistency
* Translating high-frequency visual textures into symbolic map representations
* Achieving stable GAN training with limited paired data

The dataset consists of **~1500 paired satellite–map image pairs**, enabling supervised image translation.

---

## 🧠 Approach & Methodology

We adopt a **pix2pix-style Conditional GAN**, which is a well-established and production-proven framework for paired image translation tasks.

### Core Components

* **Generator**: U-Net–based encoder–decoder with skip connections for structure preservation
* **Discriminator**: PatchGAN discriminator for enforcing local realism

### Learning Objective

The model is trained using a **multi-term loss function**:

* **Adversarial Loss** – encourages realistic map generation
* **L1 Reconstruction Loss** – enforces pixel-level alignment with ground truth
* **SSIM-based loss** – improves structural similarity

This combination balances **visual realism** and **quantitative accuracy**, making the system suitable for both academic evaluation and applied use.

---

## 🧪 Evaluation Strategy

The model is evaluated using both **quantitative** and **qualitative** methods:

### Quantitative Metrics

* **PSNR (Peak Signal-to-Noise Ratio)**
* **SSIM (Structural Similarity Index)**

### Qualitative Analysis

* Side-by-side comparison of:

  * Input satellite image
  * Generated map image
  * Ground-truth map image

This dual evaluation approach aligns with industry standards for generative vision systems.

---

## 🛠️ Technologies & Tools

### Core Frameworks
* **Python**
* **PyTorch** – deep learning framework
* **Torchvision** – datasets, transforms, and utilities

### Computer Vision & ML Concepts
* Conditional GANs (cGAN)
* Image-to-Image Translation
* Adversarial Learning
* U-Net Architecture
* PatchGAN Discriminator
* Reconstruction Loss (L1)
* Adversarial Loss
* Structural Similarity (SSIM)
* Peak Signal-to-Noise Ratio (PSNR)

### Experimentation & Development
* **OpenCV** – post-processing and image enhancement
* **PIL (Pillow)** – image loading and preprocessing
* **Matplotlib** – qualitative visualization
* **tqdm** – training and evaluation progress tracking

### Data Handling & Evaluation
* **Custom PyTorch Dataset & DataLoader**
* **Random / OS utilities** – sampling and file management
* **scikit-image** – SSIM / PSNR computation

### UI & Deployment
* **Gradio** – interactive web-based model demo
* **Flask** – backend API for handcrafted UI
* **HTML / CSS / JavaScript** – frontend interface

### Development Environment
* **Jupyter Notebook**
* **CUDA** – GPU acceleration
* **Conda / Virtual Environments**


---

```

satellite2map_cgan/
├── data/
│   ├── raw/
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/
│   ├── generator.py
│   ├── discriminator.py
│   └── pretrained/
│
├── training/
│   ├── train_baseline.py
│   └── train_cgan.py
│
├── evaluation/
│   ├── metrics.py
│   └── visualize.py
│
├── preprocessing/
│   └── preprocess_data.py
│
├── postprocessing/
│   └── post_process.py
│
├── utils/
│   ├── dataloader.py
│   └── helpers.py
│
├── ui/
│   └── gradio_app.py
│
├── results/
│   ├── samples/
│   ├── checkpoints/
│   └── logs/
│
├── notebooks/
│   ├── phase1_data_validation.ipynb
│   ├── phase4_baseline_results.ipynb
│   ├── phase5_cgan_results.ipynb
│   └── phase6_evaluation.ipynb
│
├── configs/
│   └── config.yaml
│
├── docs/
│   └── project.pdf
│
├── tests/
│   ├── test_generator.py
│   └── test_training_step.py
│
├── pipeline.ipynb
├── DESIGN.md
├── README.md
├── Development-Phases.md
├── requirements.txt
├── .gitignore
└── LICENSE


```

---

## 🚀 Use Cases & Applications

* Digital mapping and cartography
* Geospatial data visualization
* Urban planning and infrastructure analysis
* Remote sensing and satellite data interpretation
* AI-assisted map generation pipelines

---

## 📈 Project Highlights

* End-to-end **production-style GAN pipeline**
* Quantitative evaluation using industry-standard metrics
* Clean, modular, and extensible codebase
* Strong focus on **model stability and reproducibility**

---

## 👤 Author

**Mehdy Mokhtari**
Data Scientist | Computer Vision & Generative AI Engineer



