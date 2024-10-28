# $\epsilon$-Softmax: Approximating One-Hot Vectors for Mitigating Label Noise

This repository is the official **pytorch** implementation of the [eps-softmax](https://openreview.net/pdf?id=vjsd8Bcipv) [NeurIPS2024].


## How to use
**We simplify $\epsilon$-softmax with CE and FL by ECE and EFL in the code.**

**Benchmark Datasets:** The running file is `main.py`. 
* --dataset: cifar10 | cifar100, etc.
* --loss: ECEandMAE, EFLandMAE, CE, GCE, etc.
* --noise_type: symmetric | asymmetric | dependent (instance-dependent
noise), etc.

**CE $_\epsilon$+MAE (Semi):** The running file is `main_semi.py`. 
* --dataset: cifar10 | cifar100.
* --noise_type: human (cifar-n dataset), etc.

**Real-World Datasets:** The running file is `main_real_world.py`. 
* --dataset: webvision | clothing1m.
* --loss: ECEandMAE, EFLandMAE, CE, GCE, etc.

## Examples

ECEandMAE for cifar10 0.8 symmetric noise:
```console
$ python3 main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.8 --loss ECEandMAE    
```

ECEandMAE(Semi) for cifar10 human (cifar-n dataset) worst:
```console
$ python3 main_semi.py --dataset cifar10 --noise_type human --noise_rate worst  
```

ECEandMAE for webvision:
```console
$ python3 main_real_world.py --dataset webvision --loss ECEandMAE
```


## Reference
For technical details and full experimental results, please check the paper. If you have used our method or code in your own, please consider citing:

```bibtex
@inproceedings{wang2024epsilonsoftmax,
  title={\${\textbackslash}epsilon\$-Softmax: Approximating One-Hot Vectors for Mitigating Label Noise},
  author={Jialiang, Wang and Xiong, Zhou and Deming, Zhai and Junjun, Jiang and Xiangyang, Ji and Xianming, Liu},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```

