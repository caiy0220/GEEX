# GEEX
This repository contains the implementation of the black-box explainer GEEX and necessary code for the reproduction of the main results presented in: 
> [On Gradient-like Explanation under a Black-box Setting: When Black-box Explanations Become as Good as White-box](https://arxiv.org/abs/2308.09381)\
> Yi Cai, Gerhard Wunder, *ICML'24*

If you find the content of the repository useful, please consider citing our paper as:
```
@article{cai2023gradient,
  title={On Gradient-like Explanation under a Black-box Setting: When Black-box Explanations Become as Good as White-box},
  author={Cai, Yi and Wunder, Gerhard},
  journal={arXiv preprint arXiv:2308.09381},
  year={2023}
}
```

## Overview of GEEX
![GEEX overview](https://github.com/caiy0220/GEEX/raw/main/overview.png)
To derive feature attributions, GEEX prepares queries $\{\boldsymbol{z}\}$ by overlaying masks $\boldsymbol{\epsilon}\sim \pi(\cdot|\boldsymbol{x})$ on explicand variants sampled uniformly from the straightline path between the explicand and the baseline $\boldsymbol{x}(\alpha) = \mathring{\boldsymbol{x}}-\alpha(\boldsymbol{x} - \mathring{\boldsymbol{x}})$.
With the generated queries, the explainer acquires a set of observations $\{f(\boldsymbol{x}(\alpha) + \boldsymbol{\epsilon})\}$, which allows the estimation of feature attributions. 
<!-- by: $$ \boldsymbol{\xi} = \frac{(\boldsymbol{x} - \boldsymbol{\mathring{x}})}{n^*} \circ \sum_{ \substack{\boldsymbol{\epsilon}\sim\pi(\cdot|\boldsymbol{0}) \\ \alpha\sim \mathcal{U}_{[0, 1]}} } f(\boldsymbol{x}(\alpha) + \boldsymbol{\epsilon}) \nabla_{\boldsymbol{x}}\log\pi(\boldsymbol{\epsilon}|\boldsymbol{0}) $$ -->
