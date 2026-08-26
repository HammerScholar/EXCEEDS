<div align="center">

# EXCEEDS: Extracting Complex Events via Nugget-based Grid Modeling in Scientific Domain

</div>

<h5 align=center>
  
[![arXiv](https://img.shields.io/badge/arXiv-2406.14075-b31b1b.svg)](https://arxiv.org/abs/2406.14075)
[![hf](https://img.shields.io/badge/🤗-Hugging%20Face-blue.svg)](https://huggingface.co/datasets/DataHammer/SciEvents)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-yellow)](https://github.com/HammerScholar/EXCEEDS?tab=Apache-2.0-1-ov-file#readme)
[![GitHub stars](https://img.shields.io/github/stars/HammerScholar/EXCEEDS.svg?colorA=orange&colorB=orange&logo=github)](https://github.com/HammerScholar/EXCEEDS)

</h5>

This is the repository for the paper [**EXCEEDS: Extracting Complex Events via Nugget-based Grid Modeling in Scientific Domain**](https://arxiv.org/abs/2406.14075).

## 🔥 News

- **2026 May 18**: Paper is selected as an oral paper by ACL 2026 committee. Welcome to meet us in San Diego! The oral will be held at Harbor G, Session 2, Oral Session A: Information Extraction and Retrieval 1, on Sun. July 5, 11:00-12:30.
- **2026 April 28**: Paper is updated on [arXiv](https://arxiv.org/abs/2406.14075).
- **2026 April 24**: Dataset is updated on [HuggingFace](https://huggingface.co/datasets/DataHammer/SciEvents).
- **2026 April 7**: Paper is accepted by [ACL 2026](https://2026.aclweb.org/) Main Conference.
- **2025 Nov 11:** Dataset is released on [HuggingFace](https://huggingface.co/datasets/DataHammer/SciEvents).
- **2024 Jun 20:** Paper is available on [arXiv](https://arxiv.org/abs/2406.14075).

## 📊 Release of SciEvents Dataset

You can find the released dataset in [this HuggingFace repository](https://huggingface.co/datasets/DataHammer/SciEvents). 

The data format of SciEvents can be found at [data/SciEvents/README.md](data/SciEvents/README.md).

## 🔁 Reproduction of EXCEEDS

### Dataset and Pre-trained Model

Download SciEvents from [this HuggingFace repository](https://huggingface.co/datasets/DataHammer/SciEvents). Put `train.json`, `dev.json`, and `test.json` in [data/SciEvents/](data/SciEvents/) dicectory.

Download Roberta-large from [this HuggingFace repository](https://huggingface.co/FacebookAI/roberta-large/tree/main).

### Environment

```bash
conda create -n exceeds python=3.8 -y
conda activate exceeds
```

```bash
# use CUDA 11.8 for example, check your own cuda version.
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

```bash
pip install \
  "transformers==4.30.0" \
  "numpy>=1.24,<2" \
  "tqdm>=4.65" \
  "prettytable>=3.7"
```

### Train

```bash
python main.py --config config/scievents.json --output_dir outputs
```

We provide default arguments, which can be found in [main.py](main.py) and [config/scievents.json](config/scievents.json).

### Predict

```bash
python main.py --config config/scievents.json --ckpt your_best_model.state
```

We provide a checkpoint and its training log, which can be found in [this HuggingFace repository](https://huggingface.co/DataHammer/EXCEEDS).

## 📎 Citation

If you find this repository useful for your research, please cite our paper:


```bibtex
@inproceedings{lu-etal-2026-exceeds,
    title = "{EXCEEDS}: Extracting Complex Events via Nugget-based Grid Modeling in Scientific Domain",
    author = "Lu, Yi-Fan  and
      Mao, Xian-Ling  and
      Wang, Bo  and
      Liu, Xiao  and
      Huang, Heyan",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.271/",
    doi = "10.18653/v1/2026.acl-long.271",
    pages = "5997--6022",
    ISBN = "979-8-89176-390-6",
    abstract = "It is crucial to understand a specific domain by events. Extensive event extraction research has been conducted in many domains such as news, finance, and biology. However, event extraction in scientific domain is still insufficiently supported by comprehensive datasets and tailored methods. Compared with other domains, scientific domain has two characteristics: (1) denser nuggets and events, and (2) more complex information forms. To solve the above problem, considering these two characteristics, we first construct SciEvents, a large-scale multi-event document-level dataset with a schema tailored for scientific domain. It consists of 2,508 documents and 24,381 events under multi-stage manual annotation and quality control. Then, we propose EXCEEDS, an end-to-end scientific event extraction framework by encoding dense nuggets into a grid matrix and simplifying complex event extraction as a nugget-based grid modeling task. Experiments on SciEvents demonstrate state-of-the-art performances of EXCEEDS. Both the SciEvents dataset and the EXCEEDS framework are released publicly to facilitate future research."
}
```
