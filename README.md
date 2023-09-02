# ⛲ Bayesian Flow Networks

> An unofficial implementation of [Bayesian Flow Networks](https://arxiv.org/abs/2308.07037) (BFNs) for discrete variables in JAX.

![Ternary diagram of BFNs](./ternary.svg)

## ❓ What are Bayesian Flow Networks?

BFNs are a new class of generative models that share some philosophy of diffusion models: they both try to model a complex probability distribution by learning the data distribution under various levels of corruption. [^1]
In diffusion models, a neural network acts on the space of the data itself, e.g. (in the reverse process) taking as input the noisy value of every pixel and outputting a (hopefully less noisy) estimate for every pixel.[^2]
In BFNs, the neural network acts on the space of _parametrised probability distributions_ for each factorised component of the data, e.g. each pixel intensity is parametrised by the mean and standard deviation of a Gaussian distribution, and the neural network inputs and outputs estimates of the means and standard deviations for each pixel.

Whereas during the generative process diffusion models start with an image that consists of pure noise, BFNs start with a uniform prior over the individual parameters of each pixel's probability distribution.
In each step during training, the model gets to view a corrupted version of each pixel (with the level of corruption pre-determined according to some set noise schedule), and the pixel parameters are updated according to the rules of Bayesian inference.
The neural network then gets another go at estimating the parameters of the pixel distributions. These steps until barely any noise is being added to the true image, much like in diffusion models.
Conceptually, with BFN there is no need to have in mind a forward diffusion process whose reverse we are trying to match: we are just starting with prior beliefs about parameters, then updating our beliefs during the "reverse" process according to a combination of Bayesian inference and a neural network.

[^1]: The success of diffusion models shows that this is seemingly easier than trying to learn the data distribution directly.
[^2]: That estimate is often reparametrised as an estimate for the noise added to a clean image at that time-step.

### 😏 Why are they so interesting for generative models of discrete variables?

Acting on the parameters of a factorised probability distribution allows a consistent framework for modelling both continuous and discrete variables (and discretised continuous variables).
In one case the parameters will be the mean and standard deviation of a Gaussian distribution, in the other case the parameters are the logits of a categorical distribution.
Corrupting data can always be interpreted the same way: smoothing out each probability distribution through convolution with a noise kernel, then sampling from the resulting distribution.
Hence for discrete variables, there is no need to define a Markov transition kernel or a map to a continuous embedding space.

It also turns out that in all cases we are indirectly trying to maximise the log-likelihood of the data through the evidence lower bound (ELBO).[^3]

[^3]: In the paper they first present this ELBO as the expected number of bits required for Alice, who has access to the true data, to transmit it to Bob according to the scheme described above.
In this interpretation Alice sends latent variables---increasingly revealing noisy observations of the true data---to Bob, who continually updates his posterior belief of the factorised distribution according to Bayesian inference and a neural network.
The estimate for the number of bits assumes that Alice is sending latent variables then finally the true data according to an efficient _bits-back_ encoding scheme.

## 😃 Examples

The Bayesian Flow Network [preprint](https://arxiv.org/abs/2308.07037) is quite heavy on the setup needed to derive closed-form expressions for the loss and sampling procedures, but the final expressions and pseudocode are comparatively simple to implement.

Below are some notebooks that interactively demonstrate some concepts in the paper.

### 📊 [Visualising Bayesian Flow for Discrete Variables](./Visualising_Flow.ipynb)

### ⚖️ [Training a Simple Discrete "Diffusion" Model over Text](./BFN_Experiment.ipynb)


## 📁 Repository

### 🗂️ Structure

- [`/bfn/`](./bfn/)
  - [`/discrete/`](./bfn/discrete/)
    - [`training.py`](./bfn/training.py)
    - [`train_and_sample.py`](./bfn/train_and_sample.py)
    - [`example_data.py`](./bfn/example_data.py)
    - [`visualising_flows.py`](./bfn/visualising_flows.py)
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
- [ ] Bayesian flow visualisation for discrete distribution.
- [ ] Loss function and sampling for continuous probability distribution.
