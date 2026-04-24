<div align="center">

# EXCEEDS: Extracting Complex Events via Nugget-based Grid Modeling in Scientific Domain

</div>

<h5 align=center>
  
[![arXiv](https://img.shields.io/badge/arXiv-2406.14075-b31b1b.svg)](https://arxiv.org/abs/2406.14075)
[![hf](https://img.shields.io/badge/🤗-Hugging%20Face-blue.svg)](https://huggingface.co/datasets/Ralston/SciEvents)
[![License](https://img.shields.io/badge/Code%20License-Apache%202.0-yellow)](https://github.com/HammerScholar/EXCEEDS?tab=Apache-2.0-1-ov-file#readme)
[![GitHub stars](https://img.shields.io/github/stars/HammerScholar/EXCEEDS.svg?colorA=orange&colorB=orange&logo=github)](https://github.com/HammerScholar/EXCEEDS)

</h5>

This is the repository for the paper [**EXCEEDS: Extracting Complex Events via Nugget-based Grid Modeling in Scientific Domain**](https://arxiv.org/abs/2406.14075).

## 🔥 News

- **2026 April 24**: Dataset is updated on HuggingFace.
- **2026 April 7**: Paper is accepted by [ACL 2026](https://2026.aclweb.org/) Main Conference.
- **2025 Nov 11:** Dataset is released on [HuggingFace](https://huggingface.co/datasets/Ralston/SciEvents).
- **2024 Jun 20:** Paper is available on [arXiv](https://arxiv.org/abs/2406.14075).

## 📊 Release of SciEvents Dataset

You can find the released dataset in [this HuggingFace repository](https://huggingface.co/datasets/Ralston/SciEvents). 

The data format of SciEvents can be found at [data/SciEvents/README.md](data/SciEvents/README.md).

## 🔁 Reproduction of EXCEEDS

### Dataset and Pre-trained Model

Download SciEvents from [this HuggingFace repository](https://huggingface.co/datasets/Ralston/SciEvents). Put `train.json`, `dev.json`, and `test.json` in [data/SciEvents/](data/SciEvents/) dicectory.

Download Roberta-large from this [HuggingFace repository](https://huggingface.co/FacebookAI/roberta-large/tree/main).

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

We provide a checkpoint and its training log, which can be found in this HuggingFace repository.

## 📎 Citation

If you find this repository useful for your research, please cite our paper:


```bibtex
@misc{lu2024exceedsextractingcomplexevents,
      title={EXCEEDS: Extracting Complex Events as Connecting the Dots to Graphs in Scientific Domain}, 
      author={Yi-Fan Lu and Xian-Ling Mao and Bo Wang and Xiao Liu and Heyan Huang},
      year={2024},
      eprint={2406.14075},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.14075}, 
}
```
