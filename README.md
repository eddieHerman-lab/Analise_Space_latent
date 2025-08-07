# Analise_Space_latent

# Microscopic Analysis of the Latent Space: An XAI Framework

[![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

### Setup and Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/eddieHerman-lab/Analise_Space_latent.git](https://github.com/eddieHerman-lab/Analise_Space_latent.git)
    cd Analise_Space_latent
    ```
2.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

#### Part 1: Synthetic Environment

To run the analysis on the synthetic dataset, you can execute the main script. The code will generate a synthetic dataset and run the full analysis pipeline.
```bash
python src/main.py

 ```
![Descrição do Gráfico](caminho/para/o/seu_grafico_final.png)

### Performance Table

Comparative results of hyperparameter optimization, demonstrating the superiority of the "Optimized Configuration" (Run 4).
| Configuração | TP | FP | FN | Precisão | Recall | **F1-Score** |
| :--- | :--: | :--: | :--: | :---: | :---: | :---: |
| Run 1 (Super Rigoroso, P95) | 2 | 1 | 198 | 66.67% | 1.00% | **1.97%** |
| Run 2 (Rigoroso, P80) | 8 | 2 | 192 | 80.00% | 4.00% | **7.62%** |
| Run 3 (Equilibrado, P65)| 3 | 5 | 197 | 37.50% | 1.50% | **2.88%** |
| **Run 4 (Otimizado, P45)**| **15**| **0** | **185**| **100.00%**| **7.50%**| **13.95%**|


# Part 2: CelebA Case Study
## The main analysis on real-world data is contained in the Jupyter Notebook notebook_analise_celeba.ipynb.

Data Setup: Please follow the instructions in the data/README_data.md file to download and set up the CelebA dataset.

Pre-trained Model: Download the pre-trained model weights from [Your Google Drive Link Here] and place the file inside the pretrained_models/ directory.

Run Notebook: Open the notebook in a Jupyter or Google Colab environment and run the cells from top to bottom.

Key Results
The framework successfully identified and quantified a main stereotypical cluster (SBS=75.15%) and a creative niche (highest average CLS) in the CelebA latent space. The Heuristic Map revealed a strong positive correlation (Spearman's ρ = 0.64) between Uniqueness and Originality, defining a "Creative Path".

Citation
If you find this work useful in your research, please consider citing the preprint:

Code Snippet

@article{hermanson2025microscopic,
  title={Microscopic Analysis of the Latent Space: Heuristics for Interpretability, Authenticity, and Bias Detection in VAE Representations},
  author={Hermanson, Eduardo Augusto Pires},
  journal={Zenodo},
  year={2025},
  doi={Your_DOI_Here},
  url={https://...}
}


Acknowledgements
This work was developed with the assistance of several Artificial Intelligence tools that acted as research assistants. Language models such as Gemini (Google), Claude (Anthropic), ChatGPT (OpenAI), and DeepSeek were utilized in various stages of the process, including the generation and debugging of Python code, brainstorming methodological approaches, summarizing related articles, and rephrasing paragraphs to improve clarity and conciseness. The final responsibility for the content, analyses, and conclusions presented herein lies entirely with the author.




