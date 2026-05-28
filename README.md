# Stable Diffusion Fine-Tuning: Canine Breed Generation pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Diffusers-F9AB00.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Project Overview
This repository contains an end-to-end Machine Learning pipeline designed to fine-tune a pre-trained Latent Diffusion Model (Stable Diffusion v1.5) on a specialized subset of the Stanford Dogs Dataset. 

The primary objective is to adapt a generalized image generation model to accurately synthesize specific, highly detailed dog breeds. The project explores and implements multiple fine-tuning paradigms—including Unified Low-Rank Adaptation (LoRA), Class-Isolated LoRA, and DreamBooth—while strictly optimizing the computational pipeline to operate within the physical constraints of a consumer-tier GPU (NVIDIA GTX 1060 6GB VRAM).

### Key Features
* **Automated Data Extraction:** Scripts to dynamically ingest and format the Stanford Dogs dataset into Hugging Face-compliant metadata structures.
* **VRAM-Optimized Training:** Pipeline configured with gradient checkpointing, `fp16` mixed precision, and gradient accumulation to prevent CUDA Out-Of-Memory (OOM) failures on 6GB architectures.
* **Multi-Paradigm Execution:** Supports both single-class and multi-class LoRA parameter injection, alongside full-model DreamBooth fine-tuning.

---

## Academic Documentation & Engineering Report
**Course:** Advanced Problems in Applied Computer Science – Geoinformatics II  
**Evaluation Target:** 5.0 Grade Framework  

### 1. System Architecture and Training Pipeline
The system is built to ingest raw image-text pairs, map specific visual attributes to text prompts, and compute optimized weight updates.
1. **Extraction & Filtering:** Raw images of 5 visually distinct canine categories (Dachshund, Pembroke Welsh Corgi, Pug, Samoyed, and Chow Chow) are extracted from the ImageNet-indexed directory tree.
2. **Prompt Serialization:** For standard text-to-image fine-tuning, images are paired with structured target sequences following a rigid template: `"photo of {breed_name} dog"`. These are serialized into a `metadata.jsonl` file.
3. **Execution Engine:** The training framework leverages the Hugging Face `accelerate` engine, decoupling hardware configurations from pure PyTorch implementation details.

During training, images are transformed into a lower-dimensional latent space via a Variational Autoencoder (VAE). A forward diffusion process applies random Gaussian noise to these latents. The U-Net core predicts the injected noise field based on text token embeddings, calculating the Mean Squared Error (MSE) loss against the true noise vector to backpropagate weight adjustments.

### 2. Generative Models
Diffusion models (such as Stable Diffusion) generate data by formalizing image creation as an iterative denoising process over a parameterized Markov chain. A neural network learns to invert structural degradation, predicting the exact noise offset to reconstruct data out of pure randomness.

**Alternative Approaches:**
* **Generative Adversarial Networks (GANs):** Rely on a competitive minimax architecture (Generator vs. Discriminator). While offering fast single-pass generation, they suffer from training instability and mode collapse.
* **Variational Autoencoders (VAEs):** Compress data into a continuous latent space and decode it. They are mathematically stable but often produce blurrier, averaged outputs compared to the high-fidelity results of diffusion models.

### 3. Stable Diffusion Architecture
Stable Diffusion is a Latent Diffusion Model (LDM). Instead of executing calculations in high-dimensional pixel space, it operates within a highly compressed latent space.
* **Variational Autoencoder (VAE):** Compresses an input image from pixel space down into a lower-dimensional latent manifold, reducing spatial dimensions by a factor of 8.
* **CLIP Text Encoder:** Tokenizes and processes text inputs using a Vision-Language transformer. The final hidden states serve as contextual vectors.
* **Denoising U-Net:** A 2D U-Net configured with ResNet blocks and Cross-Attention layers. The Cross-Attention mechanism blends the textual context vectors into the spatial latent states to predict the noise residual.

### 4. Deep Transfer Learning and Layer Freezing
Deep Transfer Learning relies on the principle that features learned on generalized datasets (e.g., LAION-5B) can be adapted to specialized tasks. Early network layers capture universal visual features (edges, lighting), while later layers interpret high-level domain concepts.

**Layer Freezing** locks down specific weight tensors by setting their gradient flags to non-trainable (`requires_grad=False`). This prevents *catastrophic forgetting* (losing general image composition knowledge) and drastically reduces VRAM usage. In this pipeline, the VAE and CLIP Text Encoder are entirely frozen, restricting gradient updates exclusively to the Cross-Attention sub-layers of the U-Net.

### 5. Low-Rank Adaptation (LoRA)
LoRA is an efficiency-focused mathematical framework for fine-tuning large pre-trained models. Instead of performing dense matrix updates ($\Delta W$), LoRA freezes the original weights ($W_0$) and injects trainable rank decomposition matrices ($A$ and $B$) into the Transformer architecture.

The final weight is calculated as $W_{new} = W_0 + \Delta W$. By keeping the rank $r$ low, the total number of trainable parameters is reduced by over 98%, cutting hardware memory demands exponentially while maintaining high adaptability to new styles and subjects.

### 6. Method Comparison: LoRA vs. DreamBooth
Both methodologies were tested within the project scope:
* **LoRA:** Injects small, low-rank matrices into attention blocks. It is storage-efficient (~100MB) and operates comfortably within the 6GB VRAM constraint. It is best used for rapid iteration and stylized adaptations.
* **DreamBooth:** Performs a full-parameter update across the entire U-Net structure. It binds a specific subject to a rare text token (e.g., `"sks dog"`). It provides maximum subject fidelity but requires immense VRAM (>12GB) and generates heavy checkpoint files (~4GB).

### 7. Multi-Class LoRA vs. Dedicated Individual LoRAs
To evaluate subject isolation, two LoRA structural approaches were executed:
* **Paradigm A (Unified LoRA):** A single model trained on all 5 breeds simultaneously. While storage-efficient, it introduces a high risk of cross-class contamination (feature bleeding), where attributes of one breed may inadvertently influence the generation of another.
* **Paradigm B (Breed-Isolated LoRA):** 5 independent models trained separately per breed. This guarantees absolute class isolation, focusing 100% of the model's capacity on specific breed features, albeit at the cost of higher storage overhead and prolonged total training time.

### 8. Hyperparameter Optimization & Hardware Telemetry
Hyperparameters were strictly controlled to prevent pipeline failure on a consumer-grade NVIDIA GTX 1060 (6GB VRAM).

* **Resolution (`512x512`):** Matches the native pre-training resolution of SD v1.5.
* **Batch Size (`1`) & Gradient Accumulation (`4`):** Maintained at minimum to prevent immediate CUDA OOM crashes, simulating an effective batch of 4 to stabilize gradients.
* **Mixed Precision (`fp16`) & Gradient Checkpointing:** Crucial flags that cut VRAM consumption by ~50% and clear intermediate activations from memory during the backward pass.
* **Learning Rate:** `1e-04` (LoRA) and `2e-06` (DreamBooth).

**Hardware Telemetry Inference:**
The LoRA execution operated efficiently (~8s/iteration). Conversely, the full-parameter DreamBooth execution highlighted a severe hardware boundary. Exceeding the native 6GB VRAM forced the OS to page memory over the PCIe bus into system RAM, causing a memory thrashing deadlock (execution slowed to >190s/iteration). This explicitly proves the architectural necessity of LoRA for consumer-grade hardware.

### 9. Results and Analysis


* **Base Pre-Trained Model:** Generates anatomically acceptable dogs but defaults to generic canine features, failing to capture strict breed-specific nuances.
* **Unified LoRA:** Demonstrates improved breed shapes but occasionally exhibits texture bleeding across the 5 targeted breeds.
* **Isolated LoRA:** Achieves the highest alignment with prompt criteria, perfectly rendering specific skeletal proportions (e.g., elongated torso of the Dachshund) and coat textures while maintaining high intra-class diversity (various angles and backgrounds).
* **Hardware Conclusion:** The project successfully demonstrates that resource-constrained hardware can achieve state-of-the-art generative fine-tuning through strict parameter isolation (LoRA) and VRAM management strategies.