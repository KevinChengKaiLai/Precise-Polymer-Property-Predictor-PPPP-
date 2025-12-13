# Precise Polymer Property Predictor (PPPP)

This repository benchmarks **AIMNet-X2D** and **Mamba (State Space Models)** architectures to predict five key polymer properties using the large-scale dataset from the **🏆[NeurIPS 2025 Open Polymer Prediction](https://kaggle.com/competitions/neurips-open-polymer-prediction-2025)**

**Target Properties:**
- Density
- Thermal Conductivity (Tc)
- Glass Transition Temperature (Tg)
- Radius of Gyration (Rg)
- Fractional Free Volume (FFV)



## **🧠  SOTA Model  Architectures: AIMNet and Mamba**

To tackle the multi-scale nature of polymer physics, we deployed two complementary State-of-the-Art (SOTA) architectures:

### **AIMNet-X2D (Graph-Based)**

* **Structure:** A universal, scalable Graph Neural Network (GNN) framework originally designed for modeling potential energy surfaces and chemical transfer learning.  
* **Strengths:** It excels at capturing **local chemical environments** and atomic interactions (atoms-in-molecules) with high physical consistency.  
* **Why for Polymers:** Properties like **Density** and **Thermal Conductivity (Tc)** are fundamentally driven by how polymer chains pack and interact locally. AIMNet-X2D's ability to model these short-range forces allows for precise prediction of packing-related metrics.

### **Mamba (Sequence-Based)**

* **Structure:** A Selective State Space Model (SSM) that achieves **linear-time scaling** with sequence length, overcoming the quadratic bottleneck of traditional Transformers.  
* **Strengths:** Unmatched efficiency in processing extremely long sequences while maintaining global context.  
* **Why for Polymers:** Polymers are long molecular chains where properties like **Glass Transition Temperature (Tg)** and **Radius of Gyration (Rg)** depend on the connectivity and conformation of the entire backbone. Mamba allows us to ingest full polymer SMILES strings without truncation, capturing the **long-range dependencies** that local graph convolutions might miss.



## Key Challenge: Sparse Dataset (90% Missing Labels)

The NeurIPS 2025 Open Polymer Prediction dataset exhibits extreme sparsity—most samples lack experimental labels for 80% property (Density, Tc, Tg, Rg). Standard ML models trained on small, disjoint subsets often fail to generalize. Dropping rows with missing values would discard the vast majority of our data, severely limiting model performance.

### Our Solution: Two-Stage Imputation Pipeline

1. **Stage 1 (Imputation):** Train 10 single-task models (5 AIMNet + 5 Mamba) to infer missing labels with high confidence.
2. **Stage 2 (Multi-Task Training):** Use the fully-populated dataset to train final ensemble models, capturing cross-property correlations (e.g., Density ↔ FFV).


## Repository Structure

| File | Purpose | IDE |
|------|---------|-----|
| `run_AIMNet-X2D.ipynb` | Multi-task training for all 5 properties | Google Colab |
| `run_mamba_tg.ipynb` | Mamba model specialized for Tg | Google Colab |
| `run_mamba_ffv.ipynb` | Mamba model specialized for FFV | Google Colab |
| `run_on_Kaggle.ipynb` | Offline inference kernel for competition submission | Kaggle Enviroment (Linux) |
| `results/*.csv` | Predictions and imputed datasets | N/A |


## Competition Compliance

The Kaggle kernel operates fully offline with pre-loaded dependencies and model weights, outputting a `submission.csv` formatted for weighted MAE evaluation.



## Setup
These notebooks were designed to run in Colab or Kaggle, so there's no need to set anything up in your local machine.  **Every dependency setting is already scripted in the corresponding jupyter notebook.** If you like to run locally, please refer to AIMnet-X2d and Mamba github repository. 

Data: [NeurIPS Open Polymer Prediction 2025](https://kaggle.com/competitions/neurips-open-polymer-prediction-2025)


