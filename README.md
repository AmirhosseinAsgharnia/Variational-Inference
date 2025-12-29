# Variational Inference

This repository contains small, focused examples of Variational Inference (VI) in practice. VI comes from classical Bayesian statistics and provides a practical way to train probabilistic machine-learning models. Instead of learning a single fixed value for each parameter, VI learns a distribution over the model’s weights and biases.

In standard (deterministic) neural networks, training ends with a single set of weights that never changes. In contrast, a Bayesian neural network (BNN) treats weights and biases as random variables drawn from probability distributions. During prediction, the network effectively samples plausible models, allowing it to express uncertainty in its outputs.

Bayesian machine learning still relies on observed data, just as conventional methods do. The key difference is that it also incorporates prior knowledge about the parameters before seeing the data. Since this prior knowledge may be incomplete or wrong, uncertainty is a natural and essential part of the model. Where data is limited or ambiguous, a BNN becomes less confident, and where data is abundant, that uncertainty shrinks.

The examples in this repository focus on making these ideas concrete. Each problem is kept intentionally small so the behavior of variational inference, uncertainty, and Bayesian learning can be clearly seen rather than hidden behind large models or heavy abstractions.

I asked ChatGPT for three problems to solve. These problems range from easy to difficult. I did not use ChatGPT to solve these problems for me, but to critique my own solutions.

# Tier 1:

The problem:

<div align="center">

$y=sin(x) + \varepsilon(x)$,

where $x\in[-pi, pi]$,

$\varepsilon(x)\sim\mathcal{N}(0,\sigma(x)^2)$,

$\sigma(x)=0.1 + 0.4.\mathbf{1}_{|x|>\frac{\pi}{2}}$

</div>

The results:
<div align="center">
<img width="700" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/13d1201a-15d5-444e-b7ce-b7d3a4890298" />
</div>

# Tier 2:

The problem:

<div align="center">

$y=x^3 - 0.5 x^2 + \varepsilon$,

where $x\in[-1, 1]$,

$\varepsilon(x)\sim\mathcal{N}(0,0.1^2)$,

</div>