# Analise_Space_latent

# Análise Estrutural do Espaço Latente para Avaliação de Autenticidade

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) Este repositório contém o código e os experimentos para a pesquisa "Análise Estrutural do Espaço Latente: Uma Abordagem Heurística para a Avaliação de Autenticidade em Modelos Generativos".

---

## Resumo (Abstract)

A crescente sofisticação dos modelos de IA generativa apresenta desafios significativos para a auditoria de conteúdo e a detecção de autenticidade, em grande parte devido à natureza de "caixa-preta" de seus espaços latentes. Para endereçar essa lacuna, este trabalho propõe um novo framework para a análise forense do espaço latente, que opera não como um classificador, mas como um "microscópio" para investigar as propriedades estruturais das representações. Nossa metodologia emprega um funil de heurísticas, incluindo Análise de Componentes, Originalidade Entrópica e Estabilidade Temporal Simulada, para gerar um score de validação multifacetado. Através de experimentos em um dataset sintético controlado, demonstramos que o framework pode ser calibrado para otimizar o balanço entre precisão e recall, alcançando um F1-Score de 13.95% com 100% de precisão na identificação de núcleos de identidade autêntica. Concluímos que esta abordagem de análise intrínseca oferece um caminho promissor para o desenvolvimento de ferramentas de IA Explicável (XAI) para auditoria de conteúdo e futuras aplicações em análise de criatividade.

## Principais Heurísticas do Framework

Nosso método não é um classificador monolítico, mas um pipeline de análise que avalia uma representação latente com base em diferentes "lentes":

* **Unicidade de Componentes:** Utiliza a Análise de Componentes Principais (ICA) para decompor a representação e avalia a singularidade estatística de seus componentes.
* **Originalidade Entrópica:** Mede a complexidade informacional (entropia) tanto do vetor em si quanto de sua vizinhança no espaço latente.
* **Estabilidade Temporal Simulada:** Testa a robustez da identidade da representação sob pequenas perturbações, simulando a coerência ao longo do tempo.

## Estrutura do Repositório

O projeto está organizado de forma modular para maior clareza e reusabilidade:

-   `main.py`: O script principal que orquestra e executa o pipeline de análise completo.
-   `pipeline.py`: Contém a classe `AnalysisPipeline`, que gerencia o fluxo de trabalho de treinamento e análise.
-   `model.py`: Define a arquitetura da rede neural `EyeVAE`.
-   `analysis_framework.py`: Contém as classes para cada heurística de análise (`ComponentDecomposer`, `EntropicOriginalityMeasure`, etc.).
-   `data_utils.py`: Funções para a geração de dados sintéticos.
-   `requirements.txt`: Lista de todas as dependências do projeto.

## Instalação

1.  Clone este repositório:
    ```bash
    git clone [URL_DO_SEU_REPOSITORIO]
    cd [NOME_DA_PASTA]
    ```
2.  Crie um ambiente virtual (recomendado):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # No Windows: .venv\Scripts\activate
    ```
3.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

## Como Usar

Para executar o experimento principal com os dados sintéticos, basta rodar o script principal:

```bash
python main.py

Citação
Se você achar este trabalho útil para sua pesquisa, por favor, cite nosso pré-print:

[Formato da citação BibTeX que o arXiv fornecerá]
Licença
Este projeto é licenciado sob a Licença MIT. Veja o arquivo LICENSE para mais detalhes.


