# Variational Inference

This repository contains small, focused examples of Variational Inference (VI) in practice. VI comes from classical Bayesian statistics and provides a practical way to train probabilistic machine-learning models. Instead of learning a single fixed value for each parameter, VI learns a distribution over the model’s weights and biases.

In standard (deterministic) neural networks, training ends with a single set of weights that never changes. In contrast, a Bayesian neural network (BNN) treats weights and biases as random variables drawn from probability distributions. During prediction, the network effectively samples plausible models, allowing it to express uncertainty in its outputs.

Bayesian machine learning still relies on observed data, just as conventional methods do. The key difference is that it also incorporates prior knowledge about the parameters before seeing the data. Since this prior knowledge may be incomplete or wrong, uncertainty is a natural and essential part of the model. Where data is limited or ambiguous, a BNN becomes less confident, and where data is abundant, that uncertainty shrinks.

The examples in this repository focus on making these ideas concrete. Each problem is kept intentionally small so the behavior of variational inference, uncertainty, and Bayesian learning can be clearly seen rather than hidden behind large models or heavy abstractions.
