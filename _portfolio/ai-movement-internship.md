---
title: "AI Research Intern - Multi-view Breast Cancer Detection"
excerpt: "Advanced Deep Learning for Multi-View Structural Reasoning in Mammographic Analysis using Anatomy-aware Graph Networks, achieving 78.4%–92.5% Recall@[0.5,4.0]FPI"
collection: portfolio
permalink: /portfolio/ai-movement-internship/
date: 2024-08-01
venue: 'AiMovement/UM6P, Rabat, Morocco'
---

<div align="center">

<p><strong>Research Period:</strong> Été 2024</p>

<p><strong>Institution:</strong><br>
<a href="https://www.aimovement.ma/">International Center for Artificial Intelligence of Morocco (AiMovement)</a><br>
<a href="https://www.um6p.ma/">Mohammed VI Polytechnic University (UM6P)</a>, Rabat, Morocco</p>

<p><strong>Contexte:</strong><br>
Première incursion du laboratoire dans l'imagerie médicale • Implémentation du papier "Act Like a Radiologist"</p>

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

**Contexte médical**: Le cancer du sein est la pathologie néoplasique la plus répandue chez les femmes, représentant environ 2,3 millions de nouveaux cas en 2022. La mammographie demeure l'outil de dépistage de référence, mais l'analyse traditionnelle mono-vue limite souvent la détection précoce, particulièrement dans les tissus mammaires denses.

**Approche clinique**: Les radiologues analysent naturellement plusieurs vues mammographiques simultanément (crânio-caudale CC et médio-latérale oblique MLO) pour améliorer la précision diagnostique. Cette capacité de raisonnement multi-vue constitue un avantage diagnostique majeur.

**Innovation technologique**: Ce stage de recherche s'est concentré sur l'implémentation et le raffinement du papier "Act Like a Radiologist", choisي pour son équilibre entre exigences de ressources et performances, ainsi que l'absence de code open-source disponible. Notre approche développe un système de fusion multi-vue sophistiqué utilisant les **Anatomy-aware Graph Networks (AGN)** qui émulent les patterns d'interprétation radiologique.

---

## 🎯 Objectives

1. **Réviser, implémenter et affiner** les méthodes de pointe pour la détection mono-vue et multi-vue du cancer du sein
2. **Développer un pipeline de préprocessing robuste** pour l'identification de données et de landmarks anatomiques
3. **Implémenter l'architecture AGN** avec Bipartite Graph Network (BGN) pour les correspondances intra-vue et Inception Graph Network (IGN) pour la symétrie bilatérale
4. **Optimiser les performances** sous contraintes de données limitées avec approche d'entraînement en deux étapes
5. **Établir des benchmarks comparatifs** contre les frameworks établis (MaskRCNN, DETR, YOLO) sur le dataset CBIS-DDSM

---

## ⚙️ Methods

### Architecture "Act Like a Radiologist" 

L'approche radiologique standard pour l'analyse mammographique implique:
1. **Analyse de vue individuelle** pour chaque projection mammographique
2. **Corrélation inter-vues** pour identifier les lésions correspondantes  
3. **Fusion multi-vue** pour la décision diagnostique finale ← *Réseaux de Neurones Graphiques appliqués ici*

<div align="center">
  <img src="https://raw.githubusercontent.com/ibouftini/ALR-portfolio/main/images/AGN.png" alt="AGN Architecture" width="70%">
  <p><em>Architecture générale AGN avec BGN et IGN</em></p>
</div>

### Anatomy-aware Graph Neural Network (AGN)

Notre implémentation s'appuie sur l'architecture AGN qui fonctionne en imitant la capacité de raisonnement naturel que les radiologues appliquent lors du diagnostic:

**Composants clés:**
- **Bipartite Graph Network (BGN)**: Modélise les correspondances entre vues ipsilatérales (CC et MLO du même sein)
- **Inception Graph Network (IGN)**: Exploite la symétrie bilatérale entre les seins gauche et droit
- **Pseudo-landmarks**: Points de référence anatomiquement cohérents (mamelon, muscle pectoral, contour mammaire)
- **Fusion par attention**: Mécanisme résiduel pour préserver et augmenter les caractéristiques

<div align="center">
  <img src="https://raw.githubusercontent.com/ibouftini/ALR-portfolio/main/images/maskrcnn_adaptation.png" alt="MaskRCNN Adaptation" width="75%">
  <p><em>Adaptation MaskRCNN pour l'analyse multi-vue du cancer du sein</em></p>
</div>

### Preprocessing Pipeline et Extraction d'Éléments Structurels

#### Nettoyage des Données CBIS-DDSM
Le dataset CBIS-DDSM présente plusieurs défis nécessitant un preprocessing approfondi:
- **Images miroir**: 26.7% du dataset nécessitait une correction d'orientation
- **Artefacts et bordures**: Suppression par seuillage adaptatif et cropping basé coordonnées
- **Fichiers corrompus**: Détection et correction des ROI remplacés par des masques binaires
- **Incohérences de résolution**: Normalisation des dimensions entre images et masques

#### Extraction de Landmarks Anatomiques
**Détection du contour mammaire**: Seuillage OTSU avec offset ajusté $t_{adjusted} = t^* - \alpha$ et lissage B-spline

**Détection du muscle pectoral**: 
- Vues CC: Ligne verticale approximative à l'étendue médiale
- Vues MLO: Approche multi-étapes avec CLAHE, seuillage combiné, détection de contours Canny, et transformée de Hough probabiliste

**Détection du mamelon**:
- Vues CC: Point le plus latéral du contour mammaire
- Vues MLO: Analyse de courbure avec $\kappa(u) = \frac{x'(u)y''(u) - y'(u)x''(u)}{(x'(u)^2 + y'(u)^2)^{3/2}}$

<div align="center">
  <img src="https://raw.githubusercontent.com/ibouftini/ALR-portfolio/main/images/pseudo.png" alt="Pseudo-landmarks" width="40%">
  <p><em>Génération de pseudo-landmarks: (a) Vue CC, (b) Vue MLO</em></p>
</div>

### Implémentation Technique Détaillée

#### Architecture MaskRCNN Baseline
- **Backbone**: ResNet-50 + Feature Pyramid Network (FPN) pour extraction multi-échelle
- **RPN**: Anchors optimisés par K-means - 5 tailles [4,7,8,10,12] et 3 ratios [1.5,2.5,3.6]
- **ROI Align**: Configuration 7×7 pour détection, 14×14 pour segmentation
- **Têtes de détection/masque**: Classification binaire (masse/arrière-plan) + régression de boîtes

<div align="center">
  <img src="https://raw.githubusercontent.com/ibouftini/ALR-portfolio/main/images/maskrcnn_architecture.png" alt="MaskRCNN Architecture" width="80%">
  <p><em>Architecture MaskRCNN complète avec backbone ResNet-50+FPN</em></p>
</div>

#### Stratégie d'Entraînement en 3 Étapes
Pour résoudre les problèmes de surajustement avec données limitées:
1. **Étape 1**: Backbone gelé, entraînement têtes de détection uniquement (époques 0-20)
2. **Étape 2**: Dégel partiel couches backbone de haut niveau (époques 20-40)
3. **Étape 3**: Fine-tuning end-to-end avec régularisation renforcée (époques 40-60)

#### Optimisations GPU et Augmentation
- **Augmentation probabiliste en ligne**: Albumentation avec flip horizontal, rotation, affine, distorsion élastique
- **Precision mixte**: Entraînement FP16 pour optimisation mémoire
- **Configuration SGD**: LR=0.002, momentum=0.9, decay=0.0001, scheduler step=15

---

## 🛠️ Configuration Expérimentale

### Configuration Dataset CBIS-DDSM

#### Données d'Entraînement
- **Dataset principal**: CBIS-DDSM avec 1,566 patients et 3,069 images mammographiques
- **Groupes tri-vues**: 111 groupes (87 entraînement, 24 test) après filtrage patients ≥3 mammographies
- **Vues**: Crânio-caudale (CC) et médio-latérale oblique (MLO)
- **Résolution**: 4084×3328 pixels, résolution 42.5-200 μm
- **Défi statistique**: Dataset déséquilibré masses, absence d'images entièrement saines

#### Algorithme de Groupement Multi-vue
Stratégie de groupement en trois catégories: examinée, controlatérale, et auxiliaire
```python
# Algorithme de groupement tri-vue
for each patient p in P:
    if |P[p]| < 3: continue
    for each image i in P[p]:
        ve, se = View(i), Side(i)
        C = {j: View(j) = ve AND Side(j) != se}
        A = {j: View(j) != ve AND Side(j) = se}
        if C and A: create_triad(i, c, a)
```

### Infrastructure Technique

#### Configuration Matérielle
- **GPU**: NVIDIA A100 40GB pour entraînement AGRCNN
- **Optimisations**: Précision mixte automatique, gradient clipping
- **Temps d'inférence**: MaskRCNN 79ms vs AGRCNN 432ms (5.5× plus lent)

#### Stack Logiciel
- **Framework**: PyTorch avec poids pré-entraînés ImageNet
- **Preprocessing**: Modification architecture pour images niveaux de gris
- **Évaluation**: Seuil IoU réduit à 0.2 pour cohérence avec études comparatives

---

## 📊 Results

### Comparaison Performance FROC

| Modèle | R@0.5FPI | R@1.0FPI | R@2.0FPI | R@3.0FPI | R@4.0FPI | Dataset |
|--------|----------|----------|----------|----------|----------|----------|
| **ALR MaskRCNN+FPN** | 76.0% | 82.5% | 88.7% | 90.8% | 91.4% | DDSM (2,620 img) |
| **Notre MaskRCNN+FPN** | 68.9% | 79.8% | 86.3% | 90.2% | 91.3% | CBIS-DDSM (1,560 img) |
| **Notre AGRCNN** | **78.4%** | **85.5%** | **90.1%** | **91.6%** | **92.5%** | CBIS-DDSM |

<div align="center">
  <img src="https://raw.githubusercontent.com/ibouftini/ALR-portfolio/main/images/agn_froc.png" alt="FROC Comparison" width="60%">
  <p><em>Analyse FROC comparative: MaskRCNN, YOLO, DETR et AGRCNN sur CBIS-DDSM</em></p>
</div>

### Améliorations de Performance Clés

**Métriques principales:**
- **+9.5% d'amélioration** Recall@0.5FPI par rapport au baseline MaskRCNN
- **Performance supérieure** malgré 40% de données en moins vs dataset DDSM original
- **Amélioration cohérente** sur tous les seuils FPI, particulièrement significative aux faibles FPI

### Études d'Ablation Complètes

**Analyse par composants:**

| Méthode | R@0.5FPI | R@1.0FPI | R@2.0FPI | Notes |
|---------|----------|----------|----------|-------|
| **MaskRCNN (Baseline)** | 68.9% | 79.8% | 86.3% | Détection mono-vue |
| **+ BGN uniquement** | 72.1% | 81.5% | 87.8% | Correspondances ipsilatérales |
| **+ IGN uniquement** | 71.3% | 82.2% | 88.1% | Symétrie bilatérale |
| **+ AGN (fusion originale)** | 54.2% | 63.1% | 68.9% | Mécanisme d'attention destructif |
| **+ AGN (nos modifications)** | **78.4%** | **85.5%** | **90.1%** | **Connexions résiduelles** |

**Optimisation densité pseudo-landmarks:**
- **PL(13, 17)**: 76.8% recall@0.5FPI (configuration sparse)
- **PL(22, 26)**: **78.4%** recall@0.5FPI (densité optimale) ⭐
- **PL(100, 105)**: 77.2% recall@0.5FPI (sur-paramétrisation)

**Stratégie mapping kNN:**
- **k=1 (Voronoi)**: 75.2% (voisin le plus proche uniquement)
- **k=3**: **78.4%** (contexte optimal) ⭐  
- **k=5**: 77.8% (sur-lissage des caractéristiques)

### Training Evolution

<div align="center">
  <p><em>[Training curves showing loss evolution and metric improvements across epochs]</em></p>
</div>

### Qualitative Results

<div align="center">
  <p><em>[Sample detection results showing single-view vs multi-view predictions with confidence scores]</em></p>
</div>

---

## 💬 Discussion

### Contributions Techniques Majeures

#### Solution Inspirée de ResNet
**Problème identifié**: Le mécanisme d'attention AGN original était destructif, éliminant complètement les caractéristiques apprises du MaskRCNN avec $F_{enhanced} = \sigma(F_I \mathbf{w}_I) \odot F_e$ où les valeurs d'attention approchaient systématiquement zéro.

**Notre solution résiduelle**:
```python
# Attention résiduelle avec préservation des caractéristiques
ign_spatial_features = examined_features * (2.0 * ign_attention_map)
ign_spatial_features = ign_spatial_features + 0.2 * examined_features
```
Transformation de la plage d'attention de [0,1] à [0.2,2.2] permettant suppression (attention < 0.5) ET augmentation (attention > 0.5).

<div align="center">
  <img src="https://raw.githubusercontent.com/ibouftini/ALR-portfolio/main/images/agn_results_2.png" alt="AGN Results" width="75%">
  <p><em>Résultats AGN après modifications: réduction arrière-plan/contour, amélioration région masse</em></p>
</div>

#### Entraînement Progressif en 2 Étapes
1. **Étape 1**: Pré-entraînement MaskRCNN sur données mammographiques complètes
2. **Étape 2**: Intégration AGN avec poids MaskRCNN gelés pour apprentissage relations graphiques

### Signification Clinique

- **Amélioration sensibilité**: Détection supérieure de lésions subtiles manquées par analyse mono-vue
- **Réduction faux positifs**: Prédictions robustes via consensus multi-vue  
- **Workflow inspiré radiologue**: Émulation des patterns diagnostiques d'experts

### Défis et Limitations

**Contrainte données limitées**: Seulement 111 groupes tri-vues disponibles
**Solution**: Stratégie d'entraînement progressive + connexions résiduelles

**Complexité computationnelle**: Surcoût 5.5× en temps d'inférence (432ms vs 79ms)
**Impact**: Acceptable pour pipelines de dépistage clinique où précision > vitesse

**Correspondances inter-vues**: Défi d'alignement lésions entre projections différentes
**Approche**: Pseudo-landmarks anatomiques + apprentissage correspondances par attention

### Directions de Recherche Future

**Extensions immédiates**:
- **Module de classification**: Différenciation maligne/bénigne des masses détectées
- **Optimisation temps d'inférence**: Réduction surcoût computationnel preprocessing
- **Datasets plus larges**: Extension validation sur OPTIMAM, EMBED

**Perspectives ambitieuses**: 
**Limitation fondamentale 2D**: Toutes les techniques multi-vue 2D tentent d'inférer relations 3D depuis projections 2D, où la superposition tissulaire masque la distribution réelle des lésions.

**Vision 3D future**: 
- **Tomosynthèse mammaire digitale**: Exploitation information 3D native
- **Reconstruction volumétrique**: Algorithmes synthétisant descriptions 3D depuis projections mammographiques conventionnelles
- **Résolution ambiguïtés spatiales**: Différenciation masses réelles vs tissus normaux superposés

Cette limitation pointe vers une direction future plus ambitieuse: développer de véritables capacités d'analyse 3D pour résoudre la superposition tissulaire et permettre une détection confiante.

---

## 🔗 References

[1] [Author et al. "Act Like a Radiologist: Towards Reliable Multi-view Correspondence Reasoning for Mammogram Mass Detection"](https://arxiv.org/placeholder)

[2] [Shen, L., et al. (2019). "Deep Learning to Improve Breast Cancer Detection on Screening Mammography"](https://www.nature.com/articles/s41598-019-48995-4)

[3] [Kipf, T. N., & Welling, M. (2016). "Semi-Supervised Classification with Graph Convolutional Networks"](https://arxiv.org/abs/1609.02907)

[4] [Veličković, P., et al. (2017). "Graph Attention Networks"](https://arxiv.org/abs/1710.10903)

[5] [He, K., et al. (2017). "Mask R-CNN"](https://arxiv.org/abs/1703.06870)

---