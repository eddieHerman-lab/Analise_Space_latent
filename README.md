# Analise_Space_latent

# Microscopic Analysis of the Latent Space:  Heuristic Latent-Space Forensic Framework, An XAI Framework

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![arXiv](https://img.shields.io/badge/arXiv-preprint-lightgrey.svg)]()  <!-- add arXiv / DOI after publication -->


This repository contains the official code for the paper "Microscopic Analysis of the Latent Space: Heuristics for Interpretability, Authenticity, and Bias Detection in VAE Representations".

**[Read the full Pre-Print on Zenodo (Your DOI Link Here)](https://...)**

---

### Abstract

The growing sophistication of generative AI models presents significant challenges for content auditing and authenticity detection, largely due to the "black-box" nature of their latent spaces. To address this gap, this paper proposes a new framework for the structural analysis of the latent space, which operates not as a classifier, but as a "microscope" to investigate the structural properties of the representations. Our methodology was validated on a controlled synthetic dataset and then applied to a real-world case study on the CelebA dataset, revealing the framework's dual potential as a tool for both auditing bias and discovering creative outliers.

### Key Heuristics
The framework is built on a funnel of quantitative heuristics:
* **Uniqueness:** Measures the topological distinction of a sample based on its statistical independence and spatial isolation.
* **Originality:** Quantifies informational complexity using spectral and spatial entropy.
* **Creative Latent Score (CLS):** A combined metric to navigate the creative frontier of the latent space.
* **Bias Metrics (SBS & ABI):** Scores to identify and characterize stereotypical clusters.

### Project Structure

This repository is divided into two main parts:

1.  **Synthetic Environment (`/src`):** The Python scripts (`.py`) used for the experiments on the synthetic dataset, as described in Section 4 of the paper. This includes model training, hyperparameter optimization, and heuristic validation.
2.  **CelebA Case Study (`.ipynb`):** A complete Jupyter Notebook (`notebook_analise_celeba.ipynb`) containing the full analysis pipeline applied to the CelebA dataset, as described in Section 6 of the paper.



![Descrição do Gráfico](caminho/para/o/seu_grafico_final.png)

### Performance Table

Comparative results of hyperparameter optimization, demonstrating the superiority of the "Optimized Configuration" (Run 4).
| Configuration | TP | FP | FN | Precision | Recall | **F1-Score** |
| :--- | :--: | :--: | :--: | :---: | :---: | :---: |
| Run 1 (Highly Strict, P95) | 2 | 1 | 198 | 66.67% | 1.00% | **1.97%** |
| Run 2 (Strict, P80) | 8 | 2 | 192 | 80.00% | 4.00% | **7.62%** |
| Run 3 (Balanced, P65)| 3 | 5 | 197 | 37.50% | 1.50% | **2.88%** |
| **Run 4 (Optimized, P45)**| **15**| **0** | **185**| **100.00%**| **7.50%**| **13.95%**|


2 Dataset with CelebA

### Viewing the Notebook

This repository contains the source code in the `.ipynb` file. The GitHub interface may fail to render this large notebook.

**<a href="[https://nbviewer.org/github/eddieHerman-lab/Analise_Space_latent/blob/main/VAe_Ressearch_Space_latent_Analysis.ipynb]" target="_blank">Click here to view the fully rendered notebook with all figures and outputs via nbviewer.</a>**


## Quick start

## Quick start (minimal)

1. Clone:
```bash
git clone https://github.com/eddieHerman-lab/Analise_Space_latent.git
cd Analise_Space_latent

# 2) Install (recommended: venv)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3) Download data (Kaggle required):

Put your kaggle.json in ~/.kaggle/kaggle.json (chmod 600), then:

bash
Copy
Edit
bash scripts/download_data.sh
# Edit scripts/download_data.sh to set DATASET_ID first

What to expect in this repo
notebooks/ — full Colab notebook with the full pipeline and figures.

scripts/ — small helpers: download, create demo subset.

outputs/ — curated PNGs used in the preprint (kept small).

requirements.txt — pinned deps for reproducibility.

LICENSE, CITATION.cff, .gitignore

Reproducibility tips
Do not commit kaggle.json or any model checkpoints.

If you want identical figures, use the PNGs in outputs/ or fix the normalization bounds and seeds in the notebook plotting cells (set_xlim, set_ylim, fixed min/max values).

Outputs: The main paper figures are stored in outputs/images/ — run the notebook only if you want to reproduce the pipeline"


Set seeds in the notebook for NumPy/PyTorch/UMAP for more stable visuals.

Cite
When using this work, please cite the preprint (fill DOI/URL after publication):
Hermanson, E. (2025). Microscopic analysis of latent spaces: SBS/ABI/CLS heuristics. Preprint. <DOI/URL>

Contact
For collaboration or questions: eddieHerman-lab@ (or open an issue).

yaml
Copiar
Editar

---

## 2) `.gitignore` (cole na raiz)

Credentials and API keys
.kaggle/
kaggle.json
.env
*.env

Data and outputs
data/
datasets/
outputs/
results/
models/
checkpoints/
*.pth
*.ckpt
*.h5
*.pt

Notebook checkpoints
.ipynb_checkpoints

OS / Python
.DS_Store
pycache/
*.py[cod]

Virtual env
.venv/
venv/

Logs
logs/
*.log

yaml
Copiar
Editar

---

## 3) `requirements.txt` (cole na raiz)

numpy==1.25.2
scipy==1.11.2
pandas==2.2.2
matplotlib==3.8.1
seaborn==0.12.2
scikit-learn==1.3.2
torch==2.1.0
torchvision==0.16.0
hdbscan==0.8.34
umap-learn==0.5.4
moviepy==1.1.3
Pillow==10.0.1
tqdm==4.66.1
kaggle==1.5.17
ipython==8.18.0

yaml
Copiar
Editar

> Nota: se você for rodar em Colab com GPU, pode instalar `torch` com a roda CUDA apropriada — deixe instrução no README se quiser.

---

## 4) `CITATION.cff` 

```yaml
cff-version: 1.2.0
message: "If you use this code, please cite the preprint (Herman, 2025)."
title: "Microscopic analysis of latent spaces: SBS/ABI/CLS heuristics"
version: "v0.1.0"
doi: "<DOI-OR-ARXIV-LINK>"
authors:
  - family-names: Herman
    given-names: Eduardo
date-released: 2025-08-XX
license: "MIT"
url: "https://github.com/eddieHerman-lab/Analise_Space_latent"



Acknowledgements
This work was developed with the assistance of several Artificial Intelligence tools that acted as research assistants. Language models such as Gemini (Google), Claude (Anthropic), ChatGPT (OpenAI), and DeepSeek were utilized in various stages of the process, including the generation and debugging of Python code, brainstorming methodological approaches, summarizing related articles, and rephrasing paragraphs to improve clarity and conciseness. The final responsibility for the content, analyses, and conclusions presented herein lies entirely with the author.




