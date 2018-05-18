Regularized RBM for Feature Selection and Embedding
===

### Introduction

Directly learning the statistical dependencies between all observed variables in RBM will bring noisy information from irrelevant variables into the model. Thus, we introduce a l1-regularizer to mitigate the impact of those noisy variables specifically. To achieve this, we impose an l1 penalty on the probability 1 - P(x_i < t | x). Here t is a very small constant, which penalizes the reconstructed observed variables that are sensitive to large values. This penalty introduces a natural way to select the most important features (correspond to observed variables in RBM). A nice feature of this penalty is that the corresponding gradient can be computed easily.

Thus, given one training data x, we need to solve the following optimization problem. it This leads to our new formulation which performs the selection of observed variables for RBM:

![loglikelihood](https://github.com/meowoodie/Skywalker/tree/master/imgs/eq1.png)

We solve this optimization problem by gradient descent (note that this is a non-convex problem and gradient descent is a default approach to solve it). By introducing this penalty term, the gradients w_ij and b_i can be rewritten as follows:

![gradients](https://github.com/meowoodie/Skywalker/tree/master/imgs/eq2.png)

### How to use it

This work is based on [tensorfow-rbm](https://github.com/meownoid/tensorfow-rbm). The regRBM has remained same API as [tensorfow-rbm](https://github.com/meownoid/tensorfow-rbm) does.

### Experiment

![gradients](https://github.com/meowoodie/Skywalker/tree/master/imgs/exp1.png)

![gradients](https://github.com/meowoodie/Skywalker/tree/master/imgs/exp2.png)

### References
- [tensorfow-rbm](https://github.com/meownoid/tensorfow-rbm)
- [**Preprint** S. Zhu and Y. Xie, "Text Event Embeddings with Unsupervised Feature Selection"](https://arxiv.org/pdf/1710.10513.pdf)
- [S. Zhu and Y. Xie, "Crime incidents embedding using restricted boltzmann machines," 2018 IEEE International Conference on Acoustics, Speech and Signal Processing, 2018.](https://arxiv.org/pdf/1710.10513.pdf)
