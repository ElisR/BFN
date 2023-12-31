# ⛲ Bayesian Flow Networks

> An unofficial implementation of [Bayesian Flow Networks](https://arxiv.org/abs/2308.07037) (BFNs) for discrete variables in JAX.

![Ternary diagram of BFNs](./examples/ternary.svg)

## ❓ What are Bayesian Flow Networks?

BFNs are a new class of generative models that share some philosophy of diffusion models: they both try to model a complex probability distribution by iteratively learning the data distribution under various levels of corruption.[^1]
In diffusion models, a neural network acts on the space of the data itself, e.g. (in the reverse process) taking as input the noisy value of every pixel and outputting a (hopefully less noisy) estimate for every pixel.[^2]
In BFNs, the neural network acts on the space of _parametrised probability distributions_ for each factorised component of the data, e.g. each pixel intensity is parametrised by a mean and standard deviation of a Gaussian distribution, while inputs and outputs of the neural network are the estimated means and standard deviations for each pixel.

Whereas in the reverse process diffusion models start with an image that consists of pure noise, BFNs start with a uniform prior over the individual parameters of each pixel's probability distribution.
In each step during training, the model gets to view a corrupted version of each pixel (with the level of corruption set by the noise schedule), and the pixel parameters are updated according to the rules of Bayesian inference.
The neural network then sees all pixel distributions simultaneously and gets another go at updating the parameters of the pixel distributions (which is how pixel correlations get learnt).
These steps repeat until barely any noise is being added to the true image, much like in diffusion models.
Conceptually, with BFN there is no need to have in mind a forward diffusion process whose reverse we are trying to match: we are just starting with prior beliefs about parameters, then updating our beliefs during the "reverse" process according to a combination of Bayesian inference and a neural network.

[^1]: The success of diffusion models shows that this is seemingly easier than trying to learn the data distribution directly.
[^2]: That estimate is often reparametrised as an estimate for the noise added to a clean image at that time-step.

### 😏 Why are they so interesting for generative models of discrete variables?

Acting on the parameters of a factorised probability distribution allows a consistent framework for modelling both continuous and discrete variables (and discretised continuous variables in the middle!).
In one case the parameters are the means and standard deviations of Gaussian distributions, and in the other case the parameters are the logits of categorical distributions.
On both cases the parameters are real numbers.
Corrupting data can always be interpreted the same way: smoothing out each probability distribution through convolution with a noise kernel, then sampling from the resulting distribution.
Hence for discrete variables, there is no need to define a Markov transition kernel or map to a continuous embedding space to diffuse.

It also turns out that in all cases training is just maximising the log-likelihood of the data through the evidence lower bound (ELBO), without any auxiliary losses needed.[^3]

Furthermore, there are no restrictions placed on the architecture of the neural network because all it has to do is take as input a tensor of shape `(num_params, length)` and output a tensor of equal size.
When modelling discrete variables such as text tokens, many transformers already accept one-hot encoded tensors and output logits with that exact shape, so minimal modifications are needed to get up and running.
The BFN paper quotes a bits-per-character score on the `text8` dataset better than other discrete diffusion models like Multinomial Diffusion and D3PM.[^4] 

[^3]: In the paper they first present this ELBO as the expected number of bits required for Alice, who has access to the true data, to transmit it to Bob according to the BFN scheme described above.
In this interpretation Alice sends latent variables—increasingly revealing noisy observations of the true data—to Bob, who continually updates his posterior belief of the factorised distribution according to Bayesian inference and a neural network.
The estimate for the number of bits assumes that Alice sends latent variables and finally the true data according to an efficient _bits-back_ encoding scheme.
[^4]: More parameters seem to have been used than for the quoted D3PM result, so more direct comparisons would be nice.

## 😃 Examples

The Bayesian Flow Network [preprint](https://arxiv.org/abs/2308.07037) is quite heavy on the setup needed to derive closed-form expressions for the loss and sampling procedures, but the final expressions and pseudocode are comparatively simple to implement.

Below are some notebooks that interactively demonstrate some concepts in the paper.

### 📊 [Visualising Bayesian Flow for Discrete Variables](./examples/Visualising_Discrete_Flow.ipynb)

### ⚖️ [Training a Simple Discrete "Diffusion" Model on Text Characters](./examples/Discrete_BFN_Training.ipynb)

### 📈 [Visualising Bayesian Flow for Continuous Variables](./examples/Visualising_Continuous_Flow.ipynb)

### 📉 [Training a Simple Continuous "Diffusion" Model on Single Variables](./examples/Continuous_BFN_Training.ipynb)

## 📁 Repository

### ⚙️ Installation

I found it easiest to play around with this repository with `pip` installing the package in editable mode:

```bash
git clone https://github.com/ElisR/BFN.git
cd BFN
pip install -e .
```

### 🗂️ Structure

- [`/bfn/`](./bfn/)
  - [`/discrete/`](./bfn/discrete/)
    - [`training.py`](./bfn/discrete/training.py)
    - [`loss_and_sample.py`](./bfn/discrete/loss_and_sample.py)
    - [`example_data.py`](./bfn/discrete/example_data.py)
    - [`visualising_flows.py`](./bfn/discrete/visualising_flows.py)
  - [`/continuous/`](./bfn/continuous/)
  - [`utils.py`](./bfn/utils.py)
- [`/tests/`](./tests/)
  - [`/discrete/`](./tests/discrete/)
    - [`test_categorical.py`](./tests/test_categorical.py)
    - [`test_training.py`](./tests/test_training.py)
    - [`test_example_data.py`](./tests/test_example_data.py)
  - [`/continuous/`](./tests/continuous/)

### 🎯 TODOs

- [x] Loss function and sampling for discrete distributions.
- [x] Simple discrete training example notebook.
- [ ] Basic tests for discrete case.
- [x] Bayesian flow visualisation for discrete distribution.
- [x] Bayesian flow visualisation for continuous distribution.
- [x] Loss function and sampling for continuous probability distribution.
- [x] Create a working `pyproject.toml`.
