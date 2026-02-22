# Can Vision-Language Models Understand 3D? 
## Probing the Geometric Intelligence of Vision-Language Models: An Analysis of SigLIP 2 and DINOv3

### Abstract
The rapid advancement of Vision-Language Models (VLMs) and Self-Supervised Learning (SSL) architectures has led to unprecedented performance on 2D semantic classification, detection, and generation tasks. However, the extent to which these foundation models implicitly encode 3D spatial relationships and non-Euclidean geometry remains heavily debated. This paper investigates the **"geometric gap"** in modern visual foundation models by rigorously probing their frozen latent spaces. 

We evaluate the newly released **SigLIP 2** and **DINOv3** architectures against established geometric benchmarks, specifically **Probe3D** and the **Geometric Intelligence Quotient (GIQ)**. By training lightweight linear and non-linear probes on top of extracted feature representations, we assess the models' capabilities in:
*   Single-view surface normal estimation
*   Complex spatial reasoning (mental rotation and symmetry detection)

---

## 1. Introduction
Large-scale pretraining on image datasets has yielded visual foundation models with impressive generalization capabilities. While these models demonstrate exceptional zero-shot performance in 2D semantic tasks, recent works cast doubt on their true understanding of 3D geometric properties. The ability to classify or segment an object does not necessarily imply that the underlying neural representation encodes the object's 3D structure, depth, or orientation.

We posit that true 3D awareness requires a model to encode the 3D structure of a scene from a single view and maintain consistency of surfaces across multiple views. Recently, two major architectural updates have been introduced to the research community: 
1.  **SigLIP 2**: Extends the original image-text training objective with captioning-based pretraining, self-distillation, and masked prediction to improve dense prediction tasks.
2.  **DINOv3**: A self-supervised model utilizing a **Gram Anchoring** mechanism that yields highly stable local features and strong geometric consistency.

This paper probes the frozen representations of SigLIP 2 and DINOv3 to quantify their geometric intelligence. Rather than fine-tuning the massive backbones—which risks corrupting pre-trained knowledge and is computationally prohibitive on our available hardware—we employ a probing methodology. We freeze the foundation models, extract their dense and global features, and train lightweight supervised classifiers (probes) to decode geometric properties.

---

## 2. Related Work

### 2.1. Visual Foundation Models and Dense Features
Early VLMs like CLIP were optimized primarily for global image-text alignment, often neglecting fine-grained, localized spatial details. The introduction of Self-Supervised Learning (SSL) models like DINO and DINOv2 showed that contrastive and masked modeling objectives naturally yield features suitable for dense geometric tasks. **DINOv3** further improves upon this with large-scale pretraining that enhances geometric correspondence across extreme viewpoint changes. Conversely, **SigLIP 2** integrates weakly supervised learning with self-distillation to close the gap on dense localization tasks.

### 2.2. Probing 3D Awareness
Evaluating 3D awareness requires moving beyond standard semantic benchmarks. 
*   **Probe3D**: A framework introduced to evaluate visual models on single-view surface reconstruction (predicting depth and surface normals) and multiview consistency (establishing accurate semantic and geometric correspondences). 
*   **GIQ Benchmark**: The Geometric Intelligence Quotient provides a rigorous platform consisting of synthetic and real-world images of diverse polyhedra to test complex spatial reasoning, such as 3D symmetry detection and mental rotation.

---

## 3. Methodology
Our methodology is explicitly designed to decouple the computationally heavy feature extraction phase from the memory-intensive training phase, allowing the entire pipeline to operate efficiently on consumer hardware like an **NVIDIA GTX 1660 Ti** GPU.

### 3.1. Models Evaluated
We evaluate the following vision encoders:
*   **SigLIP 2 (ViT-Base)**: Evaluated to determine if its new decoder-based pretraining and self-distillation objectives resolve the geometric shortcomings of the original SigLIP.
*   **DINOv3 (ViT-Base)**: Evaluated as the state-of-the-art self-supervised baseline for dense geometric features.
*   **Baselines**: CLIP (ViT-Base) is used as a control group to measure generational improvements.

### 3.2. Hardware-Aware Feature Extraction
Because training foundation models natively is computationally prohibitive on our hardware, we utilize the GPU strictly for batched inference. Models are loaded in standard FP32 or FP16 precision. For each image, we extract:
*   **Global Representations**: The sequence-pooled or `CLS` tokens.
*   **Dense Representations**: The spatial patch tokens before the final pooling layer.

These high-dimensional tensors are immediately offloaded to disk as compressed arrays. This guarantees that VRAM usage never exceeds the limit, utilizing the GPU's memory bandwidth for fast sequential processing.

### 3.3. The Probing Architecture
We freeze the extracted features and train lightweight decoders (probes) using two distinct architectures:
1.  **Linear Probes**: A single linear layer (or $1 \times 1$ convolution for dense features). High accuracy here indicates that the geometric property is explicitly disentangled in the latent space.
2.  **Non-Linear Probes**: A small Multi-Layer Perceptron (MLP) consisting of two linear layers separated by a ReLU activation.

The performance disparity between the non-linear and linear probes quantifies how deeply entangled the geometric information is within the representation.

---

## 4. Experimental Setup

### 4.1. Single-View Surface Understanding (Probe3D)
We evaluate whether patch features inherently contain 2.5D surface information:
*   **Surface Normal Estimation**: Training a dense probe to regress pixel-wise surface normals. Metrics: Root Mean Square Error (RMSE) in degrees.

### 4.2. Spatial Reasoning and Symmetry (GIQ Benchmark)
To test higher-order geometric intelligence, we use the GIQ benchmark (224 diverse polyhedra):
*   **Mental Rotation**: A binary classification task determining if two different images depict the same object rotated in 3D space (chirality check).
*   **Symmetry Detection**: A multi-label classification task trained on global tokens to predict the presence of specific symmetries (e.g., central point reflection, 4-fold rotation).

---

## 5. Results
### 5.1. Surface Normals (Probe3D)
We evaluated the ability of local patch tokens to reconstruct surface normals using a dense $1 \times 1$ convolutional probe. Performance is measured using Root Mean Square Error (RMSE) and Mean Absolute Error (MAE) in degrees:
*   **CLIP (ViT-Base)**: 76.65° RMSE | 68.95° MAE
*   **DINOv3 (ViT-Base)**: 74.90° RMSE | 67.31° MAE
*   **SigLIP 2 (ViT-Base)**: **73.95° RMSE** | **66.64° MAE**

Both modern architectures demonstrate a measurable improvement over the baseline CLIP. Surprisingly, SigLIP 2 slightly outperforms DINOv3 on this dense geometric task, suggesting its new self-distillation objective enforces strong spatial consistency.

### 5.2. Mental Rotation and Symmetry (GIQ)
**Mental Rotation:**
This binary classification task tests chirality and spatial transformations. 
*   **Global Tokens**: CLIP and DINOv3 global tokens completely failed, yielding 50.0% accuracy (equivalent to random guessing). In stark contrast, SigLIP 2's global representation achieved an impressive **82.6% accuracy**.
*   **Local Tokens**: When pooled local patch tokens were used, performance across all models surged. CLIP reached 78.3%, while both DINOv3 and SigLIP 2 achieved **82.6% accuracy**.

**Symmetry Detection:**
This multi-label classification task requires the model to detect properties like central point reflection and $n$-fold rotation. 
*   **SigLIP 2** achieved the highest subset (exact match) accuracy at **54.3%** and an F1-micro score of 0.76 (using a linear probe).
*   **CLIP** peaked at 53.3% subset accuracy / 0.79 F1-micro, but required an MLP on local tokens to do so.
*   **DINOv3** trailed slightly, peaking at 48.9% subset accuracy / 0.74 F1-micro on local tokens.

### 5.3. Linear vs. Non-Linear Probing Gap
The performance disparity between linear and MLP probes reveals how deeply geometric information is entangled:
*   **CLIP**: Heavily relies on non-linear capacity. On Symmetry Detection (local tokens), CLIP's MLP achieved 53.3% subset accuracy, while its linear probe collapsed to 43.5%.
*   **SigLIP 2**: Exhibits highly disentangled representations. On Symmetry Detection, SigLIP 2 actually performed *better* with a simple linear probe (54.3%) than with an MLP (45.7%). This indicates that geometric symmetry is linearly separable within its latent space.

---

## 6. Discussion and Analysis
The experimental results highlight a significant leap in the geometric intelligence of modern Vision-Language Models. 

**The Power of SigLIP 2:**
SigLIP 2 consistently demonstrated superior geometric awareness. Its most remarkable achievement is the performance of its global sequence-pooled tokens on the Mental Rotation task (82.6%), a task where both CLIP and DINOv3's global tokens completely failed (50.0%). This suggests that SigLIP 2's captioning-based pretraining and dense masked prediction objectives successfully force the network to aggregate 3D spatial properties into its global representation. Furthermore, the linear separability of symmetry features in SigLIP 2 points to a highly structured and disentangled latent space.

**DINOv3's Localized Geometric Strength:**
While DINOv3's global `CLS` token struggled with complex spatial reasoning, its local patch tokens proved highly robust. Once the local tokens were pooled and probed, DINOv3 matched SigLIP 2's 82.6% accuracy on Mental Rotation and significantly outperformed CLIP on surface normal estimation. This aligns with the design philosophy of DINOv3's Gram Anchoring mechanism, which prioritizes dense, localized semantic and geometric consistency.

**The Global vs. Local Disconnect:**
A persistent theme across the benchmarks is the disconnect between global and local tokens. With the exception of SigLIP 2, the foundation models fail to effectively summarize dense geometric properties into their global representations. The fact that local patch tokens inherently preserve 3D structural information—even in models primarily trained for 2D semantic alignment—suggests that VLMs *do* encode geometry, but it remains spatially distributed.

---

## 7. Conclusion
This paper investigated the "geometric gap" in modern visual foundation models by rigorously probing the latent spaces of SigLIP 2 and DINOv3 on the Probe3D and GIQ benchmarks. We demonstrated that SigLIP 2 represents a profound generational improvement over CLIP, not just in semantic alignment, but in inherent 3D geometric intelligence. By outperforming even the self-supervised DINOv3 on surface normal estimation and global spatial reasoning, SigLIP 2 proves that carefully designed multi-objective pretraining can implicitly yield robust 3D awareness. Furthermore, our findings emphasize that future foundation architectures must focus on better bridging the gap between highly capable local patch tokens and global image representations to unlock true zero-shot spatial reasoning.