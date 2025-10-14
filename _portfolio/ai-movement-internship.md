---
title: "AI Research Intern - Multi-view Breast Cancer Detection"
excerpt: "Implementation and refinement of 'Act Like a Radiologist' paper using Anatomy-aware Graph Networks, achieving 78.4%–92.5% Recall@[0.5,4.0]FPI compared to 68.9%–91.3% baseline"
collection: portfolio
permalink: /portfolio/ai-movement-internship/
date: 2024-08-01
venue: 'AiMovement/UM6P, Rabat, Morocco'
---

<div align="center">

<p><strong>Research Period:</strong> Summer 2024</p>

<p><strong>Institution:</strong><br>
<a href="https://aim.um6p.ma/en/home/">International Center for Artificial Intelligence of Morocco (AiMovement)</a><br>
<a href="https://www.um6p.ma/">Mohammed VI Polytechnic University (UM6P)</a>, Rabat, Morocco</p>

<p><strong>Context:</strong><br>
First foray into medical imaging • Implementation of "Act Like a Radiologist" paper</p>

<h3>📋 Table of Contents</h3>
<p>
  <a href="#-introduction">📖 Introduction</a> •
  <a href="#-objectives">🎯 Objectives</a> •
  <a href="#️-methods">⚙️ Methods</a> •
  <a href="#-results">📊 Results</a> •
  <a href="#-discussion">💬 Discussion</a> •
  <a href="#-references">🔗 References</a>
</p>
</div>

---

## 📖 Introduction

Breast  cancer  is the most prevalent  neoplastic  pathology  in  women, accounting for an  estimated 2.3 million new cases in 2022. This report concerns reviewing, implementing, and refining cutting-edge methods for multi-view breast cancer detection using advanced deep learning techniques conducted at the International Artificial Intelligence Center of Morocco (AiMovement).

Our main focus was on the Anatomy-aware Graph Network (AGN) as it emulates radiologists' simultaneous analysis of mediolateral oblique (MLO) and craniocaudal (CC) views. The AGN architecture consists of an Inception Graph Network (IGN) to capture bilateral symmetry and a Bipartite Graph Network (BGN) to model intra-view correspondences. This paper was chosen in particular thanks to its ability to balance resource requirements and performance as well as the fact that its code is not publicly accessible.

---

## 🎯 Objectives

1. **Review, implement and refine** cutting-edge methods for single-view and multi-view breast cancer detection
2. **Develop a robust preprocessing pipeline** for data cleaning and landmark identification
3. **Implement AGN architecture** with resource-efficient training methods using a two-stage approach
4. **Achieve architectural adaptation** of the AGN that focuses on augmenting or weakening features rather than the original feature removal strategy to handle small-sized datasets
5. **Establish comparative benchmarks** against established frameworks (MaskRCNN, DETR, YOLO) on CBIS-DDSM dataset

---

## ⚙️ Methods

### "Act Like a Radiologist" Framework

The standard radiological approach for mammography analysis involves:
1. **Individual view analysis** for each mammographic projection
2. **Cross-view correlation** to identify corresponding lesions
3. **Multi-view fusion** for final diagnostic decision ← *Graph Neural Networks applied here*

<div align="center">
  <img src="/images/ALR-portfolio/AGN.png" alt="AGN Architecture" width="70%">
  <p><em>AGN overall architecture with BGN and IGN components</em></p>
</div>

### Anatomy-aware Graph Neural Network (AGN)

The MaskRCNN architecture is thoroughly explained in [our previous blog post](link-to-maskrcnn-post) covering ResNet-50+FPN backbone, RPN, ROI Align, and detection/segmentation heads. Here we build upon this baseline to implement the "Act Like a Radiologist" framework.

The Anatomy-aware Graph Neural Network (AGN) works by mimicking the natural reasoning ability that radiologists apply during diagnosis. AGN replicates the clinical process in which radiologists analyze numerous mammographic images to cross-validate results in contrast to traditional single-view detection.

As illustrated in Figure below, AGN applies this reasoning through a dual-graph architecture that processes ipsilateral and bilateral view relationships simultaneously.

<div align="center">
  <img src="/images/ALR-portfolio/AGN.png" alt="AGN Architecture" width="70%">
  <p><em>AGN overall architecture. The overall architecture is formulated as a function f parameterized by two specialized graphs</em></p>
</div>

The overall architecture is formulated as a function $f$ parameterized by two specialized graphs:

$$Y = f(F_{e}, F_{a}, F_{c}; \mathcal{G}_B, \mathcal{G}_I)$$

where $F_{e}, F_{a}, F_{c} \in \mathbb{R}^{HW \times C}$ represent feature maps from the examined view, auxiliary view, and contralateral view respectively, $\mathcal{G}_B$ denotes the bipartite graph for ipsilateral reasoning, and $\mathcal{G}_I$ represents the inception graph for bilateral analysis.

#### Pseudo-Landmarks: Anatomically Consistent Graph Nodes

Finding consistent spatial correlations across mammography views is the first step in creating meaningful graph representations. AGN introduced the concept of Pseudo-landmarks which are anatomically informed reference points that are relatively consistent across various patients and imaging conditions. The nipple, pectoral muscle line, and breast contour are three important anatomical features that are used to strategically position these landmarks.

Each landmark $v_i \in \mathcal{V}$ denotes an area with a generally uniform anatomical position across the patient group. The full set of pseudo-landmarks has three fundamental properties: individual landmarks correspond to unique anatomical regions, various landmarks indicate non-overlapping breast areas, and the collective landmark set covers the whole breast tissue.

#### Graph Node Mapping: From Spatial Features to Anatomical Representations

To use AGN, we should transform the visual features into graph representations using a consistent mapping mechanism. AGN employs a k-Nearest Neighbor (kNN) forward mapping function $\phi_k$ to project spatial features onto the discrete graph node domain.

The kNN mapping process operates through the following assignment matrix:

$$\phi_k(F, \mathcal{V}) = (Q^f)^T F$$

$$Q^f = A(\Lambda^f)^{-1}$$

$$A_{ij} = \begin{cases}
1, & \text{if $j$-th node is kNN of $i$-th pixel} \\
0, & \text{otherwise}
\end{cases}$$

where $A \in \mathbb{R}^{HW \times |\mathcal{V}|}$ establishes connections between image pixels and their nearest pseudo-landmark nodes, $\Lambda^f \in \mathbb{R}^{|\mathcal{V}| \times |\mathcal{V}|}$ is a diagonal normalization matrix with $\Lambda^f_{jj} = \sum_{i=1}^{HW} A_{ij}$, and $Q^f$ represents the normalized mapping matrix.

#### Bipartite Graph Network (BGN): Modeling Ipsilateral Correspondences

The Bipartite Graph Network constitutes the first component of AGN's dual-graph architecture, specifically designed to model the geometric and semantic relationships between ipsilateral mammographic views. The bipartite graph is characterized as $\mathcal{G}_B = (\mathcal{V}_{CC}, \mathcal{V}_{MLO}, \mathcal{E}_B)$.

<div align="center">
  <img src="/images/ALR-portfolio/bipartite.png" alt="Bipartite Graph Network" width="50%">
  <p><em>Bipartite Graph Network structure modeling ipsilateral correspondences</em></p>
</div>

Node features are extracted using the kNN mapping for each view:

$$X_e^B = \phi_k(F_{e}, \mathcal{V}_{l_{e}})$$
$$X_a^B = \phi_k(F_{a}, \mathcal{V}_{l_{a}})$$

The bipartite graph edge representation encapsulates the core innovation of BGN through a composite adjacency matrix that combines geometric and semantic relationship modeling:

$$H = H^g \odot H^s$$

where $H^g$ represents geometric constraints derived from anatomical knowledge, $H^s$ captures instance-specific semantic similarities, and $\odot$ denotes element-wise multiplication.

**Geometric Constraint Modeling**: The geometric graph $H^g$ encodes statistical co-occurrence patterns of corresponding anatomical regions across ipsilateral views. The geometric constraints are formalized through a frequency statistics matrix $\epsilon$ that accumulates co-occurrence counts across the training set, then normalized using:

$$H_{ij}^g = \frac{\epsilon_{ij}}{\sqrt{D_{i\cdot}D_{\cdot j}}}$$

where $D_{i\cdot} = \sum_{k=1} \epsilon_{ik}$ and $D_{\cdot j} = \sum_{k=1} \epsilon_{kj}$.

**Semantic Similarity Learning**: The semantic graph $H^s$ learns instance-specific correspondences based on visual feature similarities:

$$H_{ij}^s = \sigma([(X_i^{CC})^T, (X_j^{MLO})^T] w_s)$$

where $X_i^{CC}, X_j^{MLO} \in \mathbb{R}^C$ represent node features, $w_s \in \mathbb{R}^{2C}$ is the learnable fusion parameter, and $\sigma$ denotes the sigmoid activation function.

**Bipartite Graph Convolution**: Information propagation within BGN follows an augmented bipartite structure:

$$X^B = [(X^{CC})^T, (X^{MLO})^T]^T$$

$$H^B = \begin{pmatrix}
\mathbf{0} & H \\
H^T & \mathbf{0}
\end{pmatrix}$$

$$Z^B = \sigma(H^B X^B W^B)$$

where $W^B \in \mathbb{R}^{C \times C}$ represents learnable convolution parameters.

#### Inception Graph Network (IGN): Leveraging Bilateral Symmetry

The Inception Graph Network exploits bilateral symmetry by following the structure $\mathcal{G}_I = (\mathcal{V}_e \cup \mathcal{V}_c, \mathcal{E}_I)$, where $\mathcal{V}_e$ and $\mathcal{V}_c$ represent pseudo-landmark nodes from the examined and contralateral views.

<div align="center">
  <img src="/images/ALR-portfolio/ign.png" alt="Inception Graph Network" width="50%">
  <p><em>Inception Graph Network structure leveraging bilateral symmetry</em></p>
</div>

Node representations are obtained similarly:

$$X_e^I = \phi_k(F_e, \mathcal{V}_e)$$
$$X_c^I = \phi_k(F_c, \mathcal{V}_c)$$
$$X^I = [(X_e^I)^T, (X_c^I)^T]^T$$

**Multi-branch Adjacency Construction**: IGN handles geometric distortions through multi-branch adjacency matrices. For $n = |\mathcal{V}_e| = |\mathcal{V}_c|$ nodes, the adjacency matrix is:

$$\hat{J} = \begin{pmatrix}
M & J \\
J^T & M^T
\end{pmatrix}$$

where $M = I_n$ (identity matrix for self-loops) and $J_s$ connects each node to its top-$s$ nearest neighbors in the contralateral view.

**Inception Graph Convolution**: The inception architecture processes multiple neighborhood scales through parallel branches:

$$Z^I = \sigma\left(\begin{pmatrix}\hat{J}_{s_1} & \hat{J}_{s_2}\end{pmatrix} \begin{pmatrix}X^I & \mathbf{0} \\ \mathbf{0} & X^I\end{pmatrix} \begin{pmatrix}W^I_1 \\ W^I_2\end{pmatrix}\right)$$

where $W^I_1, W^I_2 \in \mathbb{R}^{C \times C}$ are branch-specific parameters, and $s_1, s_2$ represent different neighborhood sizes.

#### Correspondence Reasoning Enhancement and Feature Fusion

The final stage integrates multi-view reasoning capabilities into enhanced spatial feature representations. This process employs kNN reverse mapping to project graph representations back to spatial domain:

<div align="center">
  <img src="/images/ALR-portfolio/fusion.png" alt="Feature Fusion" width="30%">
  <p><em>Feature fusion mechanism combining BGN and IGN outputs</em></p>
</div>

**Reverse Mapping and Spatial Projection**: The kNN reverse mapping function transforms node features back to spatial features:

$$\psi_k(Z, \mathcal{V}_e) = Q^r [Z]_e$$
$$Q^r = (\Lambda^r)^{-1} A$$

where $\Lambda^r \in \mathbb{R}^{HW \times HW}$ is a diagonal matrix with $\Lambda^r_{ii} = \sum_{j=1}^{|\mathcal{V}_e|} A_{ij}$, and $[Z]_e$ selects nodes from the examined view.

The reverse mapping generates spatial features from both networks:

$$F_B = \psi_k(Z^B, \mathcal{V}_e)$$
$$F_I = \psi_k(Z^I, \mathcal{V}_e)$$

**Attention on IGN and Final Fusion**: IGN generates spatial attention weights highlighting asymmetric regions:

$$\hat{F}_I = \sigma(F_I w_I)$$

where $w_I \in \mathbb{R}^C$ produces attention weights.

The final enhanced features combine all information sources:

$$Y = [\hat{F}_I  \odot F_e, F_B] W_f^T$$

where $\hat{F}_I  \odot F_e$ represents attention-weighted examined view features, $F_B$ contains ipsilateral correspondence features, $\odot$ denotes element-wise product, and $W_f \in \mathbb{R}^{C \times 2C}$ is the learnable fusion parameter matrix.

This enhanced feature representation $Y$ maintains spatial dimensions compatible with standard detection architectures while incorporating multi-view anatomical reasoning capabilities, enabling seamless integration with existing mammographic detection systems.

<div align="center">
  <img src="/images/ALR-portfolio/maskrcnn_adaptation.png" alt="MaskRCNN Adaptation" width="75%">
  <p><em>MaskRCNN adaptation for Multi-view breast cancer detection</em></p>
</div>

### Data Preprocessing and Structural Element Extraction

#### CBIS-DDSM Dataset Challenges and Solutions

The CBIS-DDSM dataset contains images converted to the standard DICOM format with 1,566 patients and 3,069 mammographic images. Processing this dataset presents several challenges requiring comprehensive preprocessing:

**1. Orientation Standardization**: Anatomical orientation standardization is essential for multi-view preprocessing. The mismatch between the view and the orientation may disrupt the correspondence between views. The orientation correction code performs laterality-based flipping based on comparing the mean of first and last columns. Image sets requiring orientation correction represented 26.7% of the dataset.

**2. Artifacts Removal and Cropping**: Non-anatomical information is introduced during digitalization. Border and artifact removal constitutes a critical step. The implemented methodology employs adaptive thresholding followed by coordinate-based cropping.

**3. Corrupted Files Correction**: Some ROI files within the dataset were mistakenly replaced with their corresponding binary masks. Our implementation detects them based on data type and extracts the bounding box covering the largest connected component from the mask to crop the original mammogram.

**4. Resolution Discrepancies**: Some masks in the dataset had a different resolution than the original image, requiring geometric modifications applied consistently across files.

#### Structural Elements Calculation

Since the Act Like a Radiologist paper works on graphs, we need to transform images into graph nodes. The paper relied on multiple structural elements of mammograms (nipple, pectoral muscle and breast contour) to form their graph nodes.

**Breast Contour Detection**: We employ Otsu's thresholding method, which iteratively searches for the optimal threshold $t^*$ that maximizes the between-class variance:

$$t^* = \underset{t}{\arg\max}\{\omega_0(t)\omega_1(t)[\mu_0(t) - \mu_1(t)]^2\}$$

To exclude low intensity pixels close to image borders, we introduce a fixed offset: $t_{adjusted} = t^* - \alpha$. The raw contour is subsequently smoothed using B-spline interpolation with adaptive smoothing parameter:

$$s = \begin{cases}
10^7 & \text{if view = MLO} \\
100 & \text{if view = CC}
\end{cases}$$

**Pectoral Muscle Detection**: 

For CC views, the pectoral muscle is typically not visible. We approximate the pectoral boundary as a vertical line at the medial extent:
$$x_{pectoral} = \begin{cases}
\min_i x_i & \text{if side = Left} \\
\max_i x_i & \text{if side = Right}
\end{cases}$$

For MLO views, we employ a multi-stage approach:
1. ROI Definition in upper corner: $[0, 0.4w] \times [0, 0.6h]$ for Left side
2. CLAHE enhancement: $I_{CLAHE} = \text{CLAHE}(I_{ROI}, \text{clipLimit}=3.0)$
3. Combined thresholding: $T_{Combined} = T_{Otsu} \land T_{Adaptive}$
4. Canny edge detection: $E = \text{Canny}(T_{Combined}, \text{low}=50, \text{high}=150)$
5. Probabilistic Hough Transform for line detection
6. Line scoring based on length, position, and angle

**Nipple Detection**:

For CC views, the nipple is located at the lateralmost point:
$$p_{nipple} = \begin{cases}
\arg\min_i x_i & \text{if side = Right} \\
\arg\max_i x_i & \text{if side = Left}
\end{cases}$$

For MLO views, we employ contour curvature analysis in the lower lateral quadrant, computing:
$$\kappa(u) = \frac{x'(u)y''(u) - y'(u)x''(u)}{(x'(u)^2 + y'(u)^2)^{3/2}}$$

The optimal nipple location is determined by maximizing: $\text{score}(i) = |\kappa(u_i)|$

#### Graph Construction for Correspondence Analysis

**Pseudo-Landmark Definition**: To enable multi-view correspondence reasoning, we introduce pseudo-landmarks—anatomically consistent reference points within the breast that maintain relative spatial relationships across different views and patients.

**Landmark Generation Algorithm**:
1. Identify nipple position and pectoral muscle line as primary references
2. Define parallel lines at equidistant intervals between nipple and pectoral muscle  
3. Intersect these lines with breast contour
4. Place landmarks at uniform intervals along each line

**Graph Node Mapping**: Each pseudo-landmark serves as a graph node. We implement a k-Nearest Neighbor (kNN) mapping function:

$$\phi_k(F, V) = (Q_f)^T F$$

where $Q_f = A(\Lambda_f)^{-1}$ with assignment matrix:
$$A_{ij} = \begin{cases}
1 & \text{if $j$th node is among $k$ nearest nodes of $i$th pixel} \\
0 & \text{otherwise}
\end{cases}$$

<div align="center">
  <img src="/images/ALR-portfolio/pseudo.png" alt="Pseudo-landmarks" width="40%">
  <p><em>Pseudo-landmark generation: (a) CC view with generated landmarks, (b) MLO view with generated landmarks</em></p>
</div>

### Technical Implementation Details

#### MaskRCNN Baseline Architecture
- **Backbone**: ResNet-50 with Feature Pyramid Networks (FPN) for multi-scale feature extraction
- **RPN**: K-means optimized anchors - 5 sizes [4,7,8,10,12] and 3 aspect ratios [1.5,2.5,3.6]
- **ROI Align**: 7×7 configuration for detection, 14×14 for segmentation  
- **Detection/Mask Heads**: Binary classification (mass/background) + bounding box regression

<div align="center">
  <img src="/images/ALR-portfolio/maskrcnn_architecture.png" alt="MaskRCNN Architecture" width="80%">
  <p><em>Mask R-CNN architecture overview with ResNet-50+FPN backbone</em></p>
</div>

#### 3-Stage Training Strategy
To solve overfitting issues with limited data:
1. **Stage 1**: Frozen backbone, training detection heads only (epochs 0-20)
2. **Stage 2**: Partially unfreeze high-level backbone layers (epochs 20-40)
3. **Stage 3**: End-to-end fine-tuning with enhanced regularization (epochs 40-60)

#### Data Augmentation and GPU Optimizations
- **Probabilistic online augmentation**: Albumentations with horizontal flip, rotation, affine, elastic deformation
- **Mixed precision**: FP16 training for memory optimization
- **SGD Configuration**: LR=0.002, momentum=0.9, decay=0.0001, scheduler step=15

---

## 🛠️ Implementation & Experimental Setup

### CBIS-DDSM Dataset Configuration

#### Training Data
- **Primary dataset**: CBIS-DDSM with 1,566 patients and 3,069 mammographic images
- **Multi-view groups**: 111 groups (87 training, 24 test) after filtering patients with ≥3 mammograms
- **Views**: Craniocaudal (CC) and mediolateral oblique (MLO)
- **Resolution**: 4084×3328 pixels, resolution 42.5-200 μm
- **Statistical challenges**: Dataset completely imbalanced across masses, missing entirely healthy images

#### Multi-View Grouping Algorithm
Grouping strategy that splits mammography into three categories: examined, contralateral, and auxiliary:
```python
# Multi-View Grouping Algorithm
for each patient p in P:
    if |P[p]| < 3: continue
    for each image i in P[p]:
        ve, se = View(i), Side(i)
        C = {j: View(j) = ve AND Side(j) != se}
        A = {j: View(j) != ve AND Side(j) = se}
        if C and A: create_triad(i, c, a)
```

### Technical Infrastructure

#### Hardware Configuration
- **GPU**: NVIDIA A100 40GB for AGRCNN training
- **Optimizations**: Automatic mixed precision, gradient clipping
- **Inference Time**: MaskRCNN 79ms vs AGRCNN 432ms (5.5× slower)

#### Software Stack
- **Framework**: PyTorch with ImageNet pretrained weights
- **Preprocessing**: Architecture modification for grayscale images
- **Evaluation**: IoU threshold reduced to 0.2 for consistency with comparative studies

---

## 📊 Results

### FROC Performance Comparison

| Model | R@0.5FPI | R@1.0FPI | R@2.0FPI | R@3.0FPI | R@4.0FPI | Dataset |
|-------|----------|----------|----------|----------|----------|----------|
| **ALR MaskRCNN+FPN** | 76.0% | 82.5% | 88.7% | 90.8% | 91.4% | DDSM (2,620 img) |
| **Our MaskRCNN+FPN** | 68.9% | 79.8% | 86.3% | 90.2% | 91.3% | CBIS-DDSM (1,560 img) |
| **Our AGRCNN** | **78.4%** | **85.5%** | **90.1%** | **91.6%** | **92.5%** | CBIS-DDSM |

<div align="center">
  <img src="/images/ALR-portfolio/agn_froc.png" alt="FROC Comparison" width="60%">
  <p><em>Comparative FROC analysis: MaskRCNN, YOLO, DETR and AGRCNN on CBIS-DDSM test set</em></p>
</div>

### Key Performance Improvements

**Primary Metrics:**
- **+9.5% improvement** in Recall@0.5FPI compared to baseline MaskRCNN
- **Superior performance** despite 40% less data vs original DDSM dataset
- **Consistent improvement** across all FPI thresholds, particularly significant at low FPI

### Comprehensive Ablation Studies

**Component-wise Performance Analysis:**

| Method | R@0.5FPI | R@1.0FPI | R@2.0FPI | Notes |
|--------|----------|----------|----------|-------|
| **MaskRCNN (Baseline)** | 68.9% | 79.8% | 86.3% | Single-view detection |
| **+ BGN only** | 72.1% | 81.5% | 87.8% | Ipsilateral correspondences |
| **+ IGN only** | 71.3% | 82.2% | 88.1% | Bilateral symmetry |
| **+ AGN (Original fusion)** | 54.2% | 63.1% | 68.9% | Destructive attention mechanism |
| **+ AGN (Our modifications)** | **78.4%** | **85.5%** | **90.1%** | **Residual connections** |

**Pseudo-Landmark Density Optimization:**
- **PL(13, 17)**: 76.8% recall@0.5FPI (Sparse configuration)
- **PL(22, 26)**: **78.4%** recall@0.5FPI (Optimal density) ⭐
- **PL(100, 105)**: 77.2% recall@0.5FPI (Over-parameterized)

**Graph Node Mapping Strategy:**
- **k=1 (Voronoi)**: 75.2% (Nearest neighbor only)
- **k=3**: **78.4%** (Optimal context) ⭐  
- **k=5**: 77.8% (Over-smoothed features)


---

## 💬 Discussion

### Major Technical Contributions

#### ResNet-Inspired Feature Preservation Solution
**Problem identified**: The original AGN attention mechanism was destructive, permanently eliminating carefully learned MaskRCNN features with $F_{enhanced} = \sigma(F_I \mathbf{w}_I) \odot F_e$ where attention values systematically approached zero.

**Our residual solution**:
```python
# Residual attention mechanism with feature preservation
ign_spatial_features = examined_features * (2.0 * ign_attention_map)
ign_spatial_features = ign_spatial_features + 0.2 * examined_features
```
This transforms the effective attention range from [0,1] to [0.2,2.2], enabling both feature suppression (when attention < 0.5) and enhancement (when attention > 0.5).

<div align="center">
  <img src="/images/ALR-portfolio/agn_results_2.png" alt="AGN Results" width="75%">
  <p><em>AGN Features after model adjustments: background/contour reduction, mass region enhancement</em></p>
</div>

#### Progressive 2-Stage Training
1. **Stage 1**: Pre-training MaskRCNN on complete mammographic data
2. **Stage 2**: AGN integration with frozen MaskRCNN weights for learning graph relationships

### Clinical Significance

- **Improved Sensitivity**: Better detection of subtle lesions missed by single-view analysis
- **Reduced False Positives**: More robust predictions through multi-view consensus
- **Radiologist-inspired Workflow**: Mimics natural diagnostic patterns of expert radiologists

### Challenges and Limitations

**Limited Data Constraints**: Only 111 tri-view groups available
**Solution**: Progressive training strategy + residual connections

**Computational Complexity**: 5.5× inference time overhead (432ms vs 79ms)
**Impact**: Acceptable for clinical screening pipelines where accuracy > speed

**Cross-view Correspondence**: Challenge of accurately matching lesions across different projections
**Approach**: Anatomical pseudo-landmarks + learned correspondence through attention

### Future Research Directions

**Immediate Extensions**:
- **Classification module**: Our implementation only functions by detecting masses and cannot differentiate between malignant and benign lesions
- **Inference time optimization**: Reduction of computational preprocessing overhead
- **Larger datasets**: Extension validation on OPTIMAM, EMBED

**Ambitious Perspectives**: 
The fundamental flaw in all 2D multi-view techniques is that they try to infer 3D relationships from 2D projections, where tissue superposition obscures the actual distribution of lesions. Even the most advanced techniques cannot differentiate between actual masses and the normal tissue that covers them.

**3D Vision Future**: 
- **Digital Breast Tomosynthesis**: Exploitation of native 3D information
- **Volumetric reconstruction**: Algorithms synthesizing 3D descriptions from conventional mammographic projections
- **Spatial ambiguity resolution**: Differentiation of actual masses vs overlapping normal tissues

This limitation points toward a more ambitious future direction: developing true 3D analysis capabilities to resolve tissue superposition and allow for confident detection.

---

## 🔗 References

[1] Liu, Y., et al. (2020). "Act Like a Radiologist: Towards Reliable Multi-view Correspondence Reasoning for Mammogram Mass Detection". In MICCAI 2020.

[2] He, K., et al. (2017). "Mask R-CNN". In Proceedings of the IEEE international conference on computer vision.

[3] Lin, T. Y., et al. (2017). "Feature pyramid networks for object detection". In Proceedings of the IEEE conference on computer vision and pattern recognition.

[4] Kipf, T. N., & Welling, M. (2016). "Semi-Supervised Classification with Graph Convolutional Networks". arXiv preprint arXiv:1609.02907.

[5] Veličković, P., et al. (2017). "Graph Attention Networks". arXiv preprint arXiv:1710.10903.

---