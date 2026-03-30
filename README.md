# Image Denoising Project

This project studies **learning-based image denoising** by comparing two training settings:

1. **Synthetic noise training**  
   Train a denoising model on clean images with artificially added Gaussian noise.

2. **Real-noise training**  
   Train a denoising model on real noisy/clean image pairs from the **SIDD-Small sRGB** dataset.

The goal is to compare how different training data affect denoising performance on noisy images.

---

## Project Idea

Image denoising is a classical inverse problem.

\[
Y = X + N
\]

- \(X\): clean image  
- \(Y\): noisy image  
- \(N\): noise  

Given a noisy image \(Y\), the goal is to recover the clean image \(X\).

In this project, I compare:

- a model trained with **synthetic Gaussian noise**
- a model trained with **real noisy images**

using the same basic CNN architecture.

---

## Datasets

### 1. BSD500
Used as the clean image dataset for synthetic-noise experiments.

- Clean images are loaded from BSD500
- Gaussian noise is added to generate noisy inputs
- Random patches of size **128 × 128** are used for training

### 2. SIDD-Small sRGB
Used for real-noise experiments.

- Real noisy image / ground-truth clean image pairs
- Patch size: **128 × 128**
- Used to train and evaluate the real-noise denoising model

---

## Model

The model used in this project is a **simple CNN denoiser**.

It is intentionally lightweight so that the comparison focuses on the effect of the training data rather than on a very complex network design.

---

## Training Settings

### Synthetic baseline
- Dataset: BSD500
- Noise type: Gaussian noise
- Input: noisy patch
- Target: clean patch

### Real-noise baseline
- Dataset: SIDD-Small sRGB
- Input: real noisy patch
- Target: clean ground-truth patch

---

## Evaluation

The models are evaluated using:

- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- visual comparison of denoised outputs

---

## Current Results

### Real-noise experiment
Example evaluation result from `evaluate_sidd.py`:

- Average Noisy PSNR: **27.93 dB**
- Average Denoised PSNR: **30.14 dB**

This shows that the trained denoising model improves image quality on the SIDD-Small dataset.

### Synthetic vs Real-Noise Comparison
This project compares whether a model trained on synthetic Gaussian noise can generalize well to real noisy images, and how it differs from a model trained directly on real-noise data.

---

## Project Structure

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
