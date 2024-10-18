# [NeurIPS2024] $\epsilon$-Softmax: Approximating One-Hot Vectors for Mitigating Label Noise

This repository is the official **pytorch** implementation.


## How to use
The main running file is `main.py`. 
* We simplify $\epsilon$-softmax with CE and FL by ECE and EFL
* --loss: ECEandMAE, EFLandMAE, etc.
* --noise_type: symmetric | asymmetric | dependent

**Example:**

ECEandMAE for CIFAR-10 0.8 symmetric noise:
```console
$ python3 main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.8 --loss ECEandMAE    
```

ECEandMAE for CIFAR-10 0.6 instance-dependent noise:
```console
$ python3 main.py --dataset cifar10 --noise_type dependent --noise_rate 0.6 --loss ECEandMAE   
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

**Thanks:** Moreover, we thank the code implemented by  [Zhou et al.](https://github.com/hitcszx/lnl_sr).