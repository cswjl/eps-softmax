<div align="center">

# $\epsilon$-Softmax

### Approximating One-Hot Vectors for Mitigating Label Noise

**Official PyTorch Implementation · NeurIPS 2024**

[![arXiv](https://img.shields.io/badge/arXiv-Paper-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2508.02387)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](https://opensource.org/licenses/MIT)

[Overview](#overview) · [Poster](#poster) · [Usage](#usage) · [Examples](#examples) · [Citation](#citation) · [Contact](#contact)

</div>

---

<a id="overview"></a>

## ✨ Overview

This repository provides training scripts for benchmark, semi-supervised, and real-world noisy-label learning.

- **Losses:** $\epsilon$-softmax with CE or FL, combined with MAE.
- **Noise types:** symmetric, asymmetric, instance-dependent, human, and real-world.
- **Datasets:** CIFAR-10, CIFAR-100, CIFAR-N, WebVision, and Clothing1M.

> [!NOTE]
> In the code, **ECE** and **EFL** denote CE and FL with $\epsilon$-softmax, respectively. Their combinations with MAE are named `ECEandMAE` and `EFLandMAE`.

<a id="poster"></a>

## 🖼️ Poster

![ε-Softmax poster](poster.png)

<a id="usage"></a>

## 🛠️ Usage

```bash
git clone https://github.com/cswjl/eps-softmax.git && cd eps-softmax
```

| Setting | Entry point | Datasets | Noise types |
| :--- | :--- | :--- | :--- |
| Benchmark | [`main.py`](main.py) | `cifar10`, `cifar100` | `symmetric`, `asymmetric`, `dependent`, `human` |
| Semi-supervised | [`main_semi.py`](main_semi.py) | `cifar10`, `cifar100` | `symmetric`, `asymmetric`, `dependent`, `human` |
| Real-world | [`main_real_world.py`](main_real_world.py) | `webvision`, `clothing1m` | Real-world label noise |

| Argument | Description | Examples |
| :--- | :--- | :--- |
| `--dataset` | Dataset | `cifar10`, `cifar100`, `webvision`, `clothing1m` |
| `--loss` | Loss function (benchmark and real-world) | `ECEandMAE`, `EFLandMAE`, `CE`, `GCE` |
| `--noise_type` | Noise type (benchmark and semi-supervised) | `symmetric`, `asymmetric`, `dependent`, `human` |
| `--noise_rate` | Noise rate, or CIFAR-N label set with `human` | `0.8`, `worst`, `noisy100` |
| `--root` | Dataset root directory | `../data` (default) |

<!-- > [!NOTE]
> Training requires CUDA. Set `--root` to your data directory; for WebVision and Clothing1M, also update the image paths in the list files under [`datasets/`](datasets/). -->

<details>
<summary><strong>📂 Repository structure</strong></summary>

```text
eps-softmax/
├── main.py                 # Benchmark training
├── main_semi.py            # Semi-supervised training
├── main_real_world.py      # Real-world dataset training
├── losses.py               # Loss function implementations
├── config.py               # Loss and regularization configurations
├── models.py               # Network architectures
├── utils.py                # Training and evaluation utilities
└── datasets/               # Data loaders, noise annotations, and dataset lists
```

</details>

<a id="examples"></a>

## 🚀 Examples

```bash
# CIFAR-10 · 80% symmetric noise · ECE + MAE
python3 main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.8 --loss ECEandMAE

# CIFAR-10N · worst human labels · ECE + MAE (Semi)
python3 main_semi.py --dataset cifar10 --noise_type human --noise_rate worst

# WebVision · real-world noise · ECE + MAE
python3 main_real_world.py --dataset webvision --loss ECEandMAE
```

<a id="citation"></a>

## 🎓 Citation

If you find this work useful, please cite our [paper](https://openreview.net/pdf?id=vjsd8Bcipv):

```bibtex
@article{wang2024epsilon,
  title={$\epsilon$-Softmax: Approximating One-Hot Vectors for Mitigating Label Noise},
  author={Wang, Jialiang and Zhou, Xiong and Zhai, Deming and Jiang, Junjun and Ji, Xiangyang and Liu, Xianming},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={32012--32038},
  year={2024}
}
```

<a id="contact"></a>

## 📬 Contact

Questions about the paper or code? Contact **Jialiang Wang** at [cswjl@stu.hit.edu.cn](mailto:cswjl@stu.hit.edu.cn).

---

<div align="center">

**⭐ Star us on GitHub — it motivates us a lot!**

</div>
