# WGAN-GP for Synthetic Radiometric Data Generation

##  Impact of Normalization Strategies on WGAN-GP for Non-Gaussian Radiometric Data

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

##  Overview

This repository contains the implementation and analysis code for our study on the impact of normalization strategies on Wasserstein GAN with Gradient Penalty (WGAN-GP) for generating synthetic radiometric data. We systematically evaluate four normalization approaches, ranging from linear scaling to non-linear transformations, applied to a dataset of radionuclide concentrations and radiological indices.

**Our main contributions are threefold:**

1. **Systematic Comparison**: First comprehensive comparison of normalization strategies for GAN-based radiometric data generation
2. **Evaluation Framework**: Multi-level framework integrating univariate, multivariate, structural, and visual validation
3. **Practical Guidelines**: Demonstration that normalization choice impacts training stability, convergence, and consistency

##  Repository Structure

```
GAN_normalization/
├── scripts/
│ ├── metrics_and_plots.py # Main analysis script
│ ├── run.py # WGAN-GP training script
│ └── utils.py # Utility functions
├── Datas/
│ └── OriginalData.csv # Real radiometric dataset (109 samples, 9 variables)
├── sts_results/ # Training outputs (metrics and synthetic samples)
│ ├── cond_1/ # MinMax normalization
│ ├── cond_2/ # Log1p + MinMax normalization
│ ├── cond_3/ # Log1p + MinMax (inverted)
│ └── cond_4/ # QuantileTransformer + MinMax normalization
├── article_plots/ # Generated figures for publication
│ ├── figureA_univariate_epoch_.png
│ ├── figureB_multivariate_epoch_.png
│ ├── figureC_structure_epoch_.png
│ ├── figureD1_kde_epoch_.png
│ ├── figureD2_multivariate_epoch_.png
│ ├── figureE_temporal_dynamics.png
│ ├── table.csv
│ └── metrics_report.txt
├── config.yaml # Configuration file
├── relatorio_analise.txt # Detailed analysis report
└── requirements.txt # Python dependencies
```  

##  Getting Started

### Prerequisites

```bash
# Create and activate conda environment
conda create -n gan_env python=3.8
conda activate gan_env

# Install dependencies
pip install -r requirements.txt
```

Experimental Setup
Dataset

The real dataset consists of 109 samples and 9 radiometric variables:
|Variable	| Description | Unit |
|---------|---------------|--------
| ²²⁶Ra |	Radium-226 concentration |	Bq/kg |
|²³²Th	| Thorium-232 concentration |	Bq/kg
|⁴⁰K	| Potassium-40 concentration |	Bq/kg
|Raeq	| Radium equivalent activity |	Bq/kg
|Theq	| Thorium equivalent activity | 	Bq/kg
|Keq	| Potassium equivalent activity	| Bq/kg
|IG	| Gamma index	dimensionless  |
|IA	| Alpha index	dimensionless |
|IB	| Beta index	dimensionless |

---

Dataset is found in:
```bibtex
@dataset{barbosa_baygorrea_2026_radionuclide,
  author       = {Leandro Barbosa and Nancy Baygorrea},
  title        = {Radionuclide Dataset for Construction Materials},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {1.0},    
  doi          = {10.5281/zenodo.21348303},
  url          = {https://doi.org/10.5281/zenodo.21348303}
}
```
---
### Normalization Strategies

|Condition	| Strategy | Description |
|-----------|----------|-------------|
|C1 |	MinMax	|Linear scaling to [-1, 1] (baseline)|
|C2	| Log1p + MinMax |	Log transformation followed by MinMax|
|C3 |	Log1p + MinMax (inv)|	MinMax followed by Log transformation|
|C4	| QuantileTransformer + MinMax	| Quantile transformation followed by MinMax|

### WGAN-GP Architecture

    - **Generator**: 3 dense layers with BatchNorm and ReLU, output dimension = 9

    - **Critic (Discriminator)**: 3 dense layers with SpectralNorm and LeakyReLU (0.2)

    - **Latent Space**: z ∈ ℝ³² sampled from standard normal distribution

    - **Optimizer**: Adam (lr=1e-4, β₁=0.5, β₂=0.9)

    - **Gradient Penalty**: λ = 10

    - **Critic Updates**: 5 per generator update

    - **Batch Size**: 16

    - **Total Epochs**: 4500

###  Evaluation Metrics

- **Univariate Statistical Fidelity**

    - EMD (Earth Mover's Distance)

    - KLD (Kullback-Leibler Divergence)

    - JSD (Jensen-Shannon Divergence)

    - KS (Kolmogorov-Smirnov Statistic)

- **Multivariate Statistical Fidelity**

    MMD (Maximum Mean Discrepancy)

    Energy Distance

- **Structural Preservation**

    - Frobenius Difference

    - MACE (Mean Absolute Correlation Error)

    - Spearman Rank Correlation Difference

    - MI (Mutual Information) Difference

- **Visual Validation**

    - KDE (Kernel Density Estimation) plots

    - PCA (Principal Component Analysis) projections

    - Bivariate density maps

### Running the Analysis

1. Train the WGAN-GP Models

```bash

python scripts/run.py

```

- This will:

    - Train 4 models (one per normalization condition)

    - Run 7 independent training sessions per condition

    - Save metrics and synthetic samples in sts_results/

2. Generate Analysis and Figures
```bash

python scripts/metrics_and_plots.py
```

- This will:

    - Load all training data and metrics

    - Compute all evaluation metrics

    - Generate publication-ready figures in article_plots/

    - Create summary tables and reports

3. Configuration

- Edit config.yaml to customize paths and parameters:
yaml

- paths:
    - folder_project: "/path/to/project"
    - folder_datas: "Datas/"
    - folder_sts_results: "sts_results/"

- data:
  - numerical_columns:
    - '226Ra'
    - '232Th'
    - '40K'
    - 'Raeq'
    - 'Theq'
    - 'Keq'
    - 'IG'
    - 'IA'
    - 'IB'
  - conditions: [1, 2, 3, 4]
  - runs: [0, 1, 2, 3, 4, 5, 6]
  - target_epochs: [400, 2000, 4500]

## Results Summary
---

### Key Findings

- **Best Overall Performance**: C2 (Log1p+MinMax) achieved the best MMD (0.056) and Energy Distance (11.75)

- **Best Structural Preservation**: C4 (QuantileTransformer+MinMax) achieved the best Frobenius Difference (0.503) and MACE (0.035)

- **Most Consistent**: C4 showed the lowest inter-run variability (MMD IQR = 0.008)

- **Training Stability**: C4 demonstrated the most stable adversarial balance (G_loss = -2.339)

### Ranking

|Rank	| Condition	| Key Strength|
|-------|-----------|-------------|
|1 |	C2 (Log1p+MinMax)| 	Best multivariate similarity|
|2 |	C4 (Quantile+MinMax) |	Best structure preservation and consistency|
|3 | 	C3 (Log1p+MinMax inv)| 	Good evolution, moderate performance |
|4 |	C1 (MinMax) | 	Consistent underperformer|

### Generated Figures

|Figure | 	Content	| File|
|-------|-----------|-----|
|Figure A|	Univariate Statistical Fidelity (4 heatmaps)|	figureA_univariate_epoch_*.png|
|Figure B| 	Multivariate Similarity (Correlation + Boxplots)|	figureB_multivariate_epoch_*.png|
|Figure C|	Structure Preservation (4 boxplots)|	figureC_structure_epoch_*.png|
|Figure D1|	KDE Visualization (3×3 grid)|	figureD1_kde_epoch_*.png|
|Figure D2|	PCA + Bivariate Density|	figureD2_multivariate_epoch_*.png|
|Figure E|	Temporal Dynamics (G_loss, D_loss, EMD)|	figureE_temporal_dynamics.png|

#### Utility Scripts
---
```bash
scripts/utils.py
``` 

Contains helper functions for:

    - Data loading and preprocessing

    - Metric computation

    - File I/O operations

    - Configuration management
    
--- 

```bash 
scripts/run.py
``` 

Main training script with:

    - WGAN-GP model implementation

    - Training loop with gradient penalty

    - Model checkpointing

    - Progress monitoring

## Citation

If you use this code in your research, please cite:
```bibtex

@article{baygorrea2024normalization,
  title={Impact of Normalization Strategies on WGAN-GP for Non-Gaussian Radiometric Data: Balancing Stability, Fidelity, and Structural Preservation},
  author={Baygorrea, Nancy and Pacheco, Diego and Gomes, Otávio and Omar, Carlos and Barbosa, Leandro and Xavier, Ademir},
  journal={Journal of xxx},
  year={2026}
}
```  

