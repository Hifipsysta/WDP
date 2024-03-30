## Wasserstein Differential Privacy
This is the official PyTorch implementation for "[Wasserstein Differential Privacy](https://arxiv.org/pdf/2401.12436.pdf)" [AAAI 2024].

### Abstract
*Differential privacy (DP) has achieved remarkable results in the field of privacy-preserving machine learning. However, existing DP frameworks do not satisfy all the conditions for becoming metrics, which prevents them from deriving better basic private properties and leads to exaggerated values on privacy budgets. We propose Wasserstein differential privacy (WDP), an alternative DP framework to measure the risk of privacy leakage, which satisfies the properties of symmetry and triangle inequality. We show and prove that WDP has 13 excellent properties, which can be theoretical supports for the better performance of WDP than other DP frameworks. 
In addition, we derive a general privacy accounting method called Wasserstein accountant, which enables WDP to be applied in stochastic gradient descent (SGD) scenarios containing subsampling. Experiments on basic mechanisms, compositions and deep learning show that the privacy budgets obtained by Wasserstein accountant are relatively stable and less influenced by order. Moreover, the overestimation or even explosion on privacy budgets can be effectively alleviated.*

### Requirements

- torch 1.9.0

- matplotlib 3.4.1

- scipy 1.6.2

- numpy 1.19.2

### Experiments for Deep Learning

    python main_CIFAR.py

### How to Cite
~~~
@inproceedings{yang2024wasserstein,
  author       = {Chengyi Yang and
                  Jiayin Qi and
                  Aimin Zhou},
  title        = {Wasserstein Differential Privacy},
  booktitle    = {Thirty-Eighth {AAAI} Conference on Artificial Intelligence},
  pages        = {16299--16307},
  year         = {2024},
  doi          = {10.1609/AAAI.V38I15.29565},
}
