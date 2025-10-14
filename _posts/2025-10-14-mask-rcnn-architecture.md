---
title: "Mask R-CNN Architecture for Medical Imaging"
date: 2025-10-14
permalink: /posts/2025/10/mask-rcnn-architecture/
tags:
  - mask rcnn
  - computer vision
  - medical imaging
  - deep learning
  - object detection
  - segmentation
---

<style>
.highlight-box {
  background: linear-gradient(135deg, rgba(14, 161, 197, 0.1), rgba(14, 161, 197, 0.05));
  border-left: 4px solid #0ea1c5;
  padding: 20px;
  margin: 20px 0;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(14, 161, 197, 0.1);
}

.section-header {
  color: #0ea1c5;
  border-bottom: 3px solid #0ea1c5;
  padding-bottom: 10px;
  margin-bottom: 25px;
  font-weight: bold;
}

.subsection-header {
  color: #0ea1c5;
  border-left: 4px solid #0ea1c5;
  padding-left: 15px;
  margin: 25px 0 15px 0;
  font-weight: 600;
}

.subsubsection-header {
  color: #0ea1c5;
  font-weight: 600;
  margin: 20px 0 10px 0;
  font-size: 1.1em;
}

.key-stat {
  color: #0ea1c5;
  font-weight: bold;
  font-size: 1.1em;
}

.figure-caption {
  text-align: center;
  font-style: italic;
  color: #666;
  margin-top: 10px;
  margin-bottom: 25px;
}

.definition-box {
  background: rgba(14, 161, 197, 0.05);
  border: 1px solid rgba(14, 161, 197, 0.2);
  border-radius: 6px;
  padding: 15px;
  margin: 15px 0;
}

.tech-list {
  background: rgba(14, 161, 197, 0.05);
  border-radius: 8px;
  padding: 20px;
  margin: 20px 0;
}

.tech-list li {
  margin: 8px 0;
  padding: 5px;
}

.math-box {
  background: rgba(14, 161, 197, 0.03);
  border: 1px solid rgba(14, 161, 197, 0.15);
  border-radius: 6px;
  padding: 15px;
  margin: 15px 0;
  font-family: 'Courier New', monospace;
}

.responsive-figure-container {
  display: flex;
  justify-content: space-around;
  align-items: center;
  margin: 20px 0;
  flex-wrap: wrap;
}

.responsive-figure-container img {
  margin: 10px;
  max-width: 350px;
  width: 100%;
  height: auto;
}

@media (max-width: 768px) {
  .responsive-figure-container {
    flex-direction: column;
    align-items: center;
  }
  
  .responsive-figure-container img {
    max-width: 90%;
    margin: 10px 0;
  }
}
</style>

<div class="highlight-box">
<strong>Overview:</strong> This comprehensive guide explores the Mask R-CNN architecture for medical image analysis, covering backbone networks, region proposal mechanisms, ROI alignment, and detection/segmentation heads.
</div>

## Table of Contents
1. [Mask R-CNN Architecture Overview](#1-mask-r-cnn-architecture-overview)
2. [Backbone Networks](#2-backbone-networks)
   - 2.1 [ResNet-50 + FPN](#21-resnet-50--fpn)
   - 2.2 [Transformer Backbones](#22-transformer-backbones)
3. [Region Proposal Network (RPN)](#3-region-proposal-network-rpn)
4. [ROI Align](#4-roi-align)
5. [Detection and Mask Heads](#5-detection-and-mask-heads)
   - 5.1 [Detection Head Architecture](#51-detection-head-architecture)
   - 5.2 [Mask Head Architecture](#52-mask-head-architecture)

<h2 class="section-header">1. Mask R-CNN Architecture Overview</h2>

Mask R-CNN extends Faster R-CNN by adding a branch for predicting segmentation masks on each Region of Interest (RoI), in parallel with the existing branch for classification and bounding box regression. This architecture is particularly well-suited for medical imaging applications where precise localization and segmentation of anatomical structures are crucial.

<img src="/images/mask-rcnn-architecture/maskrcnn_overview.png" alt="Mask R-CNN architecture overview" width="800" style="display: block; margin: 0 auto;"/>
<div class="figure-caption">Figure 1: Mask R-CNN architecture overview showing the complete pipeline from input mammogram to final detection and segmentation outputs. Based on He et al.</div>

The architecture consists of several key components:
- **Input Processing**: Mammogram images (1024×1024×1)
- **Backbone Network**: ResNet-50 + FPN for multi-scale feature extraction
- **Region Proposal Network**: Generates object proposals
- **ROI Align**: Extracts fixed-size features from variable-size regions
- **Detection Head**: Performs classification and bounding box regression
- **Mask Head**: Generates pixel-level segmentation masks

<h2 class="section-header">2. Backbone Networks</h2>

<h3 class="subsection-header">2.1 ResNet-50 + FPN</h3>

ResNet-50 serves as the feature extraction backbone, addressing the vanishing gradient problem through residual connections. However, CNN encoders typically extract features at a single resolution level, forcing a trade-off between representing small details (microcalcifications) and larger structures (masses).

The solution is implemented through **Feature Pyramid Networks (FPN)**, which leverage the multi-scale pyramid hierarchy inherent in deep CNNs to create feature pyramids for detecting objects at different scales.

<img src="/images/mask-rcnn-architecture/fpn_architecture.png" alt="FPN architecture overview" width="900" style="display: block; margin: 0 auto;"/>
<div class="figure-caption">Figure 2: Feature Pyramid Network (FPN) architecture showing bottom-up pathway, lateral connections, and top-down pathway. Based on Lin et al.</div>

<div class="tech-list">
<strong style="color: #0ea1c5; margin-bottom: 10px; display: block;">FPN Key Components:</strong>
<ul style="list-style: none; padding-left: 0;">
<li><strong style="color: #0ea1c5;">• Bottom-up Pathway:</strong> ResNet-50 backbone with five convolutional stages (C₂ to C₅) that increase channels and decrease resolution</li>
<li><strong style="color: #0ea1c5;">• Lateral Connections:</strong> 1×1 convolutional layers that reduce channel dimensions to 256 and connect bottom-up features to top-down pathway</li>
<li><strong style="color: #0ea1c5;">• Top-Down Pathway:</strong> Creates feature maps P₂–P₅ by combining high-semantic features with precise spatial information</li>
<li><strong style="color: #0ea1c5;">• P₆ Generation:</strong> Additional level for detecting very large masses that occupy significant mammogram regions</li>
<li><strong style="color: #0ea1c5;">• Final Processing:</strong> 3×3 convolution on each pyramid level to reduce aliasing artifacts</li>
</ul>
</div>

<h3 class="subsection-header">2.2 Transformer Backbones</h3>

During implementation, we experimented with transformer-based backbones to leverage their powerful feature extraction capabilities, particularly benefiting from available pre-trained checkpoints in PyTorch ready for fine-tuning.

<div class="definition-box">
<strong style="color: #0ea1c5; display: block; margin-bottom: 15px;">Transformer Architectures Tested:</strong>
<ul style="list-style: none; padding-left: 0;">
<li style="margin: 10px 0; padding: 8px; background: rgba(14, 161, 197, 0.05); border-radius: 4px;"><strong style="color: #0ea1c5;">Vision Transformer (ViT):</strong> Pioneered transformer architectures for image processing</li>
<li style="margin: 10px 0; padding: 8px; background: rgba(14, 161, 197, 0.08); border-radius: 4px;"><strong style="color: #0ea1c5;">Data-efficient Image Transformer (DeiT):</strong> Optimized for training with minimal data and resources</li>
<li style="margin: 10px 0; padding: 8px; background: rgba(14, 161, 197, 0.12); border-radius: 4px;"><strong style="color: #0ea1c5;">Shifted Window Transformer (Swin):</strong> Outperforms ViT and DeiT on numerous vision tasks</li>
</ul>
</div>

<div class="highlight-box">
<strong style="color: #0ea1c5;">Key Finding:</strong> DeiT was the only transformer-based model that produced acceptable results due to data scarcity. However, its performance remained lower than ResNet-50, leading us to continue with the CNN backbone.
</div>

<h2 class="section-header">3. Region Proposal Network (RPN)</h2>

The Region Proposal Network identifies regions in feature maps with potential objects before refinement in the second detection stage. It processes feature maps along with user-defined anchors based on size and aspect ratio.

<div class="highlight-box">
<strong style="color: #0ea1c5;">Anchor Configuration:</strong> Using K-means optimization, we determined the best anchor parameters: <span class="key-stat">5 sizes [4,7,8,10,12]</span> and <span class="key-stat">3 aspect ratios [1.5,2.5,3.6]</span> for each FPN level.
</div>

<img src="/images/mask-rcnn-architecture/rpn_architecture.png" alt="RPN Architecture overview" width="600" style="display: block; margin: 0 auto;"/>
<div class="figure-caption">Figure 3: Region Proposal Network (RPN) architecture showing the parallel classification and regression branches. Based on He et al.</div>

The RPN applies a 3×3 convolution with ReLU activation, followed by two parallel branches:

1. **Classification Branch**: 1×1 convolution with 30 channels (15 anchors × 2 classes)
2. **Regression Branch**: 1×1 convolution with 60 channels (15 anchors × 4 coordinates)

<div class="math-box">
<strong>RPN Loss Function:</strong><br><br>
<strong>Objectness Loss (Binary Cross-Entropy):</strong><br>
$$\mathcal{L}_{\text{objectness}} = \frac{1}{N_{\text{cls}}} \sum_{i} L_{\text{cls}}(p_i, p_i^*)$$<br><br>
Where: $L_{\text{cls}}(p_i, p_i^*) = -p_i^* \log(p_i) - (1 - p_i^*) \log(1 - p_i)$<br><br>
<strong>Variables:</strong><br>
• $p_i$ = predicted objectness probability for anchor $i$<br>
• $p_i^* \in \{0, 1\}$ = ground truth label (1 = object, 0 = background)<br>
• $N_{\text{cls}}$ = batch size
</div>

<div class="math-box">
<strong>RPN Box Regression Loss (Smooth L1):</strong><br><br>
$$\mathcal{L}_{\text{reg}} = \frac{1}{N_{\text{reg}}} \sum_{i} p_i^* \cdot L_{\text{reg}}(t_i, t_i^*)$$<br><br>
Where: $L_{\text{reg}}(t_i, t_i^*) = \sum_{j \in \{x,y,w,h\}} \text{smooth}_{\ell_1}(t_{i,j} - t_{i,j}^*)$<br><br>
<strong>Smooth ℓ₁ Function:</strong><br>
$$\text{smooth}_{\ell_1}(x) = \begin{cases}
0.5x^2 & \text{if } |x| < 1 \\
|x| - 0.5 & \text{otherwise}
\end{cases}$$
</div>

<div class="definition-box">
<strong style="color: #0ea1c5; display: block; margin-bottom: 15px;">Anchor Assignment Rules:</strong>
<ul>
<li>$p_i^* = 1$ if IoU with ground truth > 0.7</li>
<li>$p_i^* = 0$ if IoU with ground truth < 0.3</li>
<li>Anchors with 0.3 ≤ IoU ≤ 0.7 are ignored during training</li>
</ul>
</div>

The total RPN loss combines both components:
$$\mathcal{L}_{\text{RPN}} = \frac{1}{N_{\text{cls}}} \sum_{i} L_{\text{cls}}(p_i, p_i^*) + \frac{1}{N_{\text{reg}}} \sum_{i} p_i^* L_{\text{reg}}(t_i, t_i^*)$$

<h2 class="section-header">4. ROI Align</h2>

ROI Align extracts fixed-size feature maps from variable-size regions of interest while preserving exact spatial alignment necessary for precise segmentation of mammographic masses.

Consider a feature tensor of size 32×32×512 from a network input image of dimensions 1×1024×1024. The scale factor is 1024/32 = 32.

<img src="/images/mask-rcnn-architecture/roi_representation.png" alt="ROI representation on feature map" width="500" style="display: block; margin: 0 auto;"/>
<div class="figure-caption">Figure 4: Representation of the predicted region on the feature map showing scale factor calculation. Based on He et al.</div>

<div class="highlight-box">
<strong style="color: #0ea1c5;">ROI Align Configuration:</strong>
<ul>
<li><strong>Detection output size:</strong> 7×7 pixels for classification/regression head</li>
<li><strong>Mask output size:</strong> 14×14 pixels for segmentation head</li>
<li><strong>Sampling ratio:</strong> 2 (4 points per bin)</li>
<li><strong>Spatial alignment:</strong> Floating-point coordinates (no quantization)</li>
</ul>
</div>

<img src="/images/mask-rcnn-architecture/roi_align_sampling.png" alt="ROI Align sampling pattern" width="600" style="display: block; margin: 0 auto;"/>
<div class="figure-caption">Figure 5: ROI Align 7×7 sampling pattern with bilinear interpolation from grid neighbors. Based on He et al.</div>

<div class="math-box">
<strong>Sampling Coordinates:</strong><br><br>
$$X_{\text{sample}} = X_{\text{bin}} + \text{bin\_width} \times (0.25 + 0.5 \times n_x)$$<br>
$$Y_{\text{sample}} = Y_{\text{bin}} + \text{bin\_height} \times (0.25 + 0.5 \times n_y)$$<br><br>
where $n_x, n_y \in \{0, 1\}$
</div>

For each sampling point at non-integer coordinates (x, y), the feature value is computed using bilinear interpolation from the four nearest integer grid points:

$$P \approx \frac{y_2 - y}{y_2 - y_1} \left( \frac{x_2 - x}{x_2 - x_1} Q_{11} + \frac{x - x_1}{x_2 - x_1} Q_{21} \right) + \frac{y - y_1}{y_2 - y_1} \left( \frac{x_2 - x}{x_2 - x_1} Q_{12} + \frac{x - x_1}{x_2 - x_1} Q_{22} \right)$$

<h2 class="section-header">5. Detection and Mask Heads</h2>

After ROI Align extracts fixed-size feature representations, these features are processed by two specialized heads for detection and segmentation tasks.

<h3 class="subsection-header">5.1 Detection Head Architecture</h3>

The detection head consists of a shared fully connected layer with <span class="key-stat">4096 neurons</span> receiving flattened ROI features (7×7×512 = 25,088 inputs), followed by two specialized branches:

<div class="tech-list">
<strong style="color: #0ea1c5; margin-bottom: 10px; display: block;">Detection Head Components:</strong>
<ul style="list-style: none; padding-left: 0;">
<li><strong style="color: #0ea1c5;">• Classification Branch:</strong> Softmax classifier with C+1 neurons (2 outputs for background and mass)</li>
<li><strong style="color: #0ea1c5;">• Regression Branch:</strong> 4C coordinates for refined box parameters per class</li>
</ul>
</div>

<div class="math-box">
<strong>Detection Loss Functions:</strong><br><br>
<strong>Classification Loss (Binary Cross-Entropy):</strong><br>
$$\mathcal{L}_{\text{cls}} = \text{Binary Cross-Entropy}$$<br><br>
<strong>Bounding Box Regression Loss (Smooth L1):</strong><br>
$$\mathcal{L}_{\text{bbox}}(t^u, v) = \sum_{i \in \{x,y,w,h\}} \text{smooth}_{\ell_1}(t^u_i - v_i)$$<br><br>
where $v$ indicates ground truth refinements and $t^u$ the predicted bounding box refinements for class $u$.
</div>

<h3 class="subsection-header">5.2 Mask Head Architecture</h3>

The mask head generates pixel-level segmentation masks through a series of convolutional layers, culminating in a 1×1 convolution that produces 14×14×1 output masks.

<div class="highlight-box">
<strong style="color: #0ea1c5;">Key Feature:</strong> The mask prediction is upscaled using bilinear interpolation to align with the original ROI dimensions, ensuring accurate spatial correspondence with the input image.
</div>

<div class="math-box">
<strong>Mask Loss Function (Pixel-wise Binary Cross-Entropy):</strong><br><br>
$$\mathcal{L}_{\text{mask}} = -\frac{1}{m^2} \sum_{1 \leq i,j \leq m} \left[ y_{ij} \log \hat{y}_{ij}^k + (1 - y_{ij}) \log(1 - \hat{y}_{ij}^k) \right]$$<br><br>
where:<br>
• $m$ = mask resolution (14 in our case)<br>
• $\hat{y}_{ij}^k$ = predicted probability at pixel $(i,j)$ for class $k$<br>
• $y_{ij}$ = ground truth binary mask value at location $(i,j)$
</div>

<div class="highlight-box">
<strong style="color: #0ea1c5;">Important Note:</strong> The mask loss is computed only for pixels within the predicted bounding box and only for the ground truth class, making the training process more efficient and focused.
</div>

---

*This blog post is part of a series on deep learning architectures for medical image analysis.*