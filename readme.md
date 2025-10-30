# Joint Diffuse Optical Tomography-Photoacoustic Tomography Reconstruction for Breast Lesion Imaging

This repository contains the full set of Python scripts and utilities developed for quantitative **Joint Diffuse Optical Tomography-Photoacoustic Tomography (DOT-PAT) imaging**, including **fluence modeling**, **joint absorption–scattering reconstruction**, **Nakagami parameter estimation**, and **quantitative hemoglobin analysis**.  

---

## Overview

This repository provides end-to-end computational tools for quantitative photoacoustic imaging and analysis for breast cancer imaging:

### **1. Fluence Modeling**
- 3D diffusion-based solver for heterogeneous tissue models with Robin boundary conditions.
- Supports adjoint-based gradient computation for joint reconstruction of scattering coefficient.

### **2. Statistical Ultrasound Analysis**
- Local Nakagami parameter estimation from envelope data.
- Supports adaptive windowing and median-based aggregation.

### **3. Joint Reconstruction**
- Solves for absorption (μₐ) and reduced scattering (μₛ′) maps using:
  $J(\mu_a, \mu_s') = \tfrac{1}{2}\|\hat{p} - p_{meas}\|_2^2 + \lambda_a\|\mu_a\|_1 + \tfrac{\lambda_s}{2}\|\nabla^2 \mu_s'\|_2^2$
- Gradient-based optimization using FISTA-style updates and convergence checks.

### **4. Quantitative Hemoglobin Estimation**
- Multi-wavelength spectral unmixing for HbO₂ and HbR.

---

## Installation

Clone this repository:

```bash
git clone https://github.com/YXLin1159/DOT-guided_PAT-reconstruction.git
cd DOT-guided-PAT-reconstruction
```

## Contact
**Yixiao Lin**  
Department of Biomedical Engineering  
Washington University in St. Louis  
lin.yixiao@wustl.edu
