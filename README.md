# Truncated Loss (GCE)
 
This is the unofficial PyTorch implementation of the paper "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels" in NIPS 2018.<br> 
https://arxiv.org/abs/1805.07836


## Overview

The code doesn't include the experiment for Fashion MNIST dataset. All hyperparameters are set the same as they are mentioned in the paper. There are two things that are different from the original paper. The first thing is that I didn't seperate a validation set for the pruning step to obtain the optimal-epoch model. I used the model from the best test accuracy epoch to conduct the pruning step which will of course result in better performance. If you want to have fair comparison with other methods, you should serperate a validation set. The second difference is that the loss was averaged instead of summed which I found it to be more stable. I didn't spend time running different niose rate settings. I simply pick noise rate 0.4 to validate on CIFAR-10 and CIFAR-100.


## Dependencies
This code is based on Python 3.5, with the main dependencies being PyTorch==1.2.0 torchvision==0.4.0 Additional dependencies for running experiments are: numpy, argparse, os, csv, sys, PIL


Run the code with the following example commands:<br>
###  Uniform Noise with noise rate 0.4 on CIFAR-10
```
$ CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.4 --schedule 40 80 --start_prune 40 --epochs 120
```
###  Class Dependent Noise with noise rate 0.4 on CIFAR-10
```
$ CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset cifar10 --noise_type pairflip --noise_rate 0.4 --schedule 40 80 --start_prune 40 --epochs 120
```
###  Uniform Noise with noise rate 0.4 on CIFAR-100

```
$ CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset cifar100 --noise_type symmetric --noise_rate 0.4 --schedule 80 120 --start_prune 80 --epochs 150
```
###  Class Dependent Noise with noise rate 0.4 on CIFAR-100

```
$ CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset cifar100 --noise_type pairflip --noise_rate 0.4 --schedule 80 120 --start_prune 80 --epochs 150
```



