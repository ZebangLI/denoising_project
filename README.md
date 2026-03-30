# Learning-Based Image Denoising: Synthetic Noise vs Real Noise

A deep learning project that compares **synthetic-noise training** and **real-noise training** for image denoising using the same CNN backbone.

This project studies a core imaging inverse problem:

\[
Y = X + N
\]

where a noisy image \(Y\) is formed from a clean image \(X\) and noise \(N\). The goal is to recover \(X\) from \(Y\).

---

## Overview

Many denoising models are trained on **synthetically corrupted clean images**, typically with Gaussian noise. However, real-world image noise is often more complex than a simple parametric model.

In this project, I built and evaluated two denoising pipelines:

- **Synthetic-noise baseline**: trained on BSD500 clean images with added Gaussian noise
- **Real-noise baseline**: trained on paired noisy/clean images from the SIDD-Small sRGB dataset

By keeping the model architecture fixed and changing only the training data source, this project analyzes how **data realism affects denoising performance and generalization**.

---

## Technical Highlights

- Built an end-to-end **image denoising pipeline** in **PyTorch**
- Implemented two training settings:
  - synthetic Gaussian-noise training
  - real-noise paired-image training
- Used a lightweight **CNN denoiser** as a controlled baseline
- Processed images into **128×128 training patches**
- Evaluated performance using:
  - **PSNR**
  - **SSIM**
  - visual output comparison
- Developed separate training / evaluation scripts for synthetic and real-noise experiments

---

## Datasets

### BSD500
Used as the clean-image source for synthetic-noise experiments.

- clean natural images
- Gaussian noise added during preprocessing
- noisy / clean patch pairs generated for training

### SIDD-Small sRGB
Used for real-noise denoising experiments.

- real smartphone noisy images
- corresponding clean ground-truth targets
- paired supervision for training and evaluation

---

## Model

The project uses a **simple CNN denoiser** as a baseline model.

This design keeps the architecture intentionally simple so that the comparison focuses on the effect of the **training data distribution** rather than on network complexity.

---

## Experiment Design

### 1. Synthetic-Noise Training
- input: clean BSD500 image + synthetic Gaussian noise
- target: original clean image
- purpose: establish a standard controlled denoising baseline

### 2. Real-Noise Training
- input: real noisy image from SIDD
- target: corresponding clean ground-truth image
- purpose: study denoising under real sensor noise conditions

### 3. Comparison Goal
Compare how the two training strategies differ in:

- denoising quality
- real-world robustness
- quantitative performance
- visual output characteristics

---

## Results

### Real-noise evaluation
From `evaluate_sidd.py`:

- **Average Noisy PSNR:** 27.93 dB
- **Average Denoised PSNR:** 30.14 dB

This shows a measurable improvement in reconstruction quality after denoising on the SIDD-Small dataset.

### Key takeaway
A model trained on real noisy-clean image pairs is better aligned with real-world denoising conditions than a model trained only on synthetic Gaussian noise. Synthetic-noise training remains useful as a clean and interpretable baseline.

---

## Repository Structure

```text
denoising_project/
├── models/
│   └── simple_cnn.py
├── utils/
│   ├── metrics.py
│   └── noise.py
├── train.py
├── train_sidd.py
├── evaluate.py
├── evaluate_sidd.py
├── save_val_results.py
├── save_sidd_results.py
├── check_cuda.py
├── test_model.py
├── test_dataset.py
├── test_dataloader.py
├── test_inference.py
├── test_sidd_dataset.py
├── test_sidd_dataloader.py
├── test_sidd_pair.py
├── test_sidd_inference.py
└── README.md
