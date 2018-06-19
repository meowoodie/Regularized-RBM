Regularized RBM for Feature Selection and Embedding
===

### Introduction

Directly learning the statistical dependencies between all observed variables in RBM will bring noisy information from irrelevant variables into the model. Thus, we introduce a *l1*-regularizer to mitigate the impact of those noisy variables specifically. To achieve this, we impose an *l1* penalty on the activation probability. Here t is a very small constant, which penalizes the reconstructed observed variables that are sensitive to large values. This penalty introduces a natural way to select the most important features (correspond to observed variables in RBM). A nice feature of this penalty is that the corresponding gradient can be computed easily.

Thus, given one training data x, we need to solve the following optimization problem. This leads to our new formulation which performs the selection of observed variables for RBM:

![loglikelihood](https://github.com/meowoodie/RegRBM/blob/master/imgs/eq1.png)

We solve this optimization problem by gradient descent (note that this is a non-convex problem and gradient descent is a default approach to solve it). By introducing this penalty term, the gradients can be rewritten as follows:

![gradients](https://github.com/meowoodie/RegRBM/blob/master/imgs/eq2.png)

### How to use it

This work is based on [tensorfow-rbm](https://github.com/meownoid/tensorfow-rbm). The regRBM has remained the same API as [tensorfow-rbm](https://github.com/meownoid/tensorfow-rbm) does.

Below is a simple example on how to train a regRBM:
```python
rbm = RegRBM(n_visible=n_x, n_hidden=1000, t=t, lam=lam, \
             learning_rate=lr, momentum=0.95, err_function="mse", \
             sample_visible=False, sigma=1.)
errs, zeros = rbm.fit(data_x, n_epoches=n_epoches, batch_size=20, \
                      shuffle=True, verbose=True)
```
where additional parameters `t` is the constant that controls the threshold for disabling variables, and `lam` is the factor of the regularization.

### Experiments

> Fitted RBM with and without designed penalty term over 2,056 crime text (including 7,038 keywords). Under same experiment settings, (a): training errors over iterations; (b): numbers of eliminated (disabled) variables over iterations, and (c): result of  cross-validation over different lambda value.

![gradients](https://github.com/meowoodie/RegRBM/blob/master/imgs/exp1.png)

> Selected features: (a): the standard deviations of tf-idf intensity over 2,056 crime text; (b): the same plot as (a) but the tf-idf intensity is reconstructed by a fitted RBM with regularization by taking the raw data as input. Top 15 keywords with the highest standard deviations have been annotated by the side of corresponding bars. The *x*-axis is the 7,038 keywords, and the *y*-axis is the standard deviations of each keyword.

![gradients](https://github.com/meowoodie/RegRBM/blob/master/imgs/exp2.png)

### References
- Paper: [**Preprint** S. Zhu and Y. Xie, "Crime Event Embeddings with Unsupervised Feature Selection"](https://arxiv.org/pdf/1806.06095.pdf)
- Paper: [S. Zhu and Y. Xie, "Crime incidents embedding using restricted boltzmann machines," 2018 IEEE International Conference on Acoustics, Speech and Signal Processing, 2018.](https://arxiv.org/pdf/1710.10513.pdf)
- Github: [tensorfow-rbm](https://github.com/meownoid/tensorfow-rbm)
