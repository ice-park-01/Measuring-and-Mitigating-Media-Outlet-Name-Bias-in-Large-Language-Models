# Measuring and Mitigating Media Outlet Name Bias in Large Language Models

This repository contains the code and data for the paper **"Measuring and Mitigating Media Outlet Name Bias in Large Language Models"** published at **EMNLP 2025**.

## 📄 Paper
[Link to Anthology](https://aclanthology.org/2025.emnlp-main.1513/)

## 📖 Overview
Recent studies have shown that Large Language Models (LLMs) exhibit political bias. This project investigates a specific type of bias: **Media Outlet Name Bias**. We analyze how LLMs' generated summaries and responses are affected by the mere presence of a media outlet's name in the prompt, even when the content remains neutral or identical.

Key features of this codebase:
- **Bias Measurement**: Tools to measure how different media outlet names shift the political leaning of LLM outputs.
- **Mitigation Strategies**: Implementation of mitigation techniques to reduce this bias.
- **Analysis**: Scripts to visualize and interpret the bias shifts.

## 🛠️ Environment Setup

Install the required dependencies using `pip`:

```bash
pip install -r requirements_xai.txt
```

**Requirements include:**
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- SHAP, Captum (for XAI analysis)
- Plotly, Kaleido, Seaborn, Matplotlib (for visualization)

## 📂 Directory Structure

| Directory | Description |
|-----------|-------------|
| `analyze/` | Scripts for analyzing experimental results and bias cases. |
| `analyze_result/` | Output directory for analysis results, including CSVs and figures. |
| `custom_dataset/` | Dataset handling, prompt generation, and media outlet lists. |
| `model/` | Inference scripts (`inference.py`) and XAI tools (`inference_xai.py`). |
| `scripts/` | Utility scripts (e.g., model downloading, token counting). |
| `shell/` | Shell scripts to run large-scale experiments and measurements. |
| `src/` | Core source code for model wrappers (`open_source.py`, `closed_source.py`). |

## 🚀 Quick Start

### 1. Running Experiments
Use the scripts in the `shell/` directory to run full experiments. For example, to test AllSides media bias with Gemma 2:

```bash
bash shell/run_allsides_gemma2.sh
```

## � Datasets

We used the following datasets for our experiments:

### 1. AllSides Dataset
```bibtex
@inproceedings{baly2020we,
  author      = {Baly, Ramy and Da San Martino, Giovanni and Glass, James and Nakov, Preslav},
  title       = {We Can Detect Your Bias: Predicting the Political Ideology of News Articles},
  booktitle   = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  series      = {EMNLP~'20},
  NOmonth     = {November},
  year        = {2020},
  pages       = {4982--4991},
  NOpublisher = {Association for Computational Linguistics}
}
```

### 2. Hyperpartisan Dataset
```bibtex
@inproceedings{kiesel-etal-2019-semeval,
    title = "{S}em{E}val-2019 Task 4: Hyperpartisan News Detection",
    author = "Kiesel, Johannes  and
      Mestre, Maria  and
      Shukla, Rishabh  and
      Vincent, Emmanuel  and
      Adineh, Payam  and
      Corney, David  and
      Stein, Benno  and
      Potthast, Martin",
    editor = "May, Jonathan  and
      Shutova, Ekaterina  and
      Herbelot, Aurelie  and
      Zhu, Xiaodan  and
      Apidianaki, Marianna  and
      Mohammad, Saif M.",
    booktitle = "Proceedings of the 13th International Workshop on Semantic Evaluation",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/S19-2145/",
    doi = "10.18653/v1/S19-2145",
    pages = "829--839",
    abstract = "Hyperpartisan news is news that takes an extreme left-wing or right-wing standpoint. If one is able to reliably compute this meta information, news articles may be automatically tagged, this way encouraging or discouraging readers to consume the text. It is an open question how successfully hyperpartisan news detection can be automated, and the goal of this SemEval task was to shed light on the state of the art. We developed new resources for this purpose, including a manually labeled dataset with 1,273 articles, and a second dataset with 754,000 articles, labeled via distant supervision. The interest of the research community in our task exceeded all our expectations: The datasets were downloaded about 1,000 times, 322 teams registered, of which 184 configured a virtual machine on our shared task cloud service TIRA, of which in turn 42 teams submitted a valid run. The best team achieved an accuracy of 0.822 on a balanced sample (yes : no hyperpartisan) drawn from the manually tagged corpus; an ensemble of the submitted systems increased the accuracy by 0.048."
}
```


## �📊 Citation
If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{park-kim-2025-measuring,
    title = "Measuring and Mitigating Media Outlet Name Bias in Large Language Models",
    author = "Park, Seong-Jin  and
      Kim, Kang-Min",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1513/",
    doi = "10.18653/v1/2025.emnlp-main.1513",
    pages = "29766--29785",
    ISBN = "979-8-89176-332-6",
    abstract = "Large language models (LLMs) have achieved remarkable performance across diverse natural language processing tasks, but concerns persist regarding their potential political biases. While prior research has extensively explored political biases in LLMs' text generation and perception, limited attention has been devoted to biases associated with media outlet names. In this study, we systematically investigate the presence of media outlet name biases in LLMs and evaluate their impact on downstream tasks, such as political bias prediction and news summarization. Our findings demonstrate that LLMs consistently exhibit biases toward the known political leanings of media outlets, with variations across model families and scales. We propose a novel metric to quantify media outlet name biases in LLMs and leverage this metric to develop an automated prompt optimization framework. Our framework effectively mitigates media outlet name biases, offering a scalable approach to enhancing the fairness of LLMs in news-related applications."
}
```
