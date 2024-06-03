

# ProDAG

Julia implementation of ProDAG from the paper [“ProDAG:
Projection-induced variational inference for directed acyclic
graphs”](https://arxiv.org/abs/2405.15167).

## Installation

To install `ProDAG` from GitHub, run the following code:

``` julia
using Pkg
Pkg.add(url = "https://github.com/ryan-thompson/ProDAG.jl")
```

## Usage

The `fit_linear()` function learns the posterior over linear DAGs given
data `x`. The `sample()` function draws DAGs from the learned posterior.

``` julia
using ProDAG, CUDA, Random

CUDA.seed!(1)
Random.seed!(1)

# Generate some data
n = 100
p = 5
x = randn(n, p)

# Fit a posterior with a N(0,1) prior on each weight
fit = fit_linear(x, prior_μ = 0, prior_σ = 1, verbose = false)

# Draw a sample of DAGs from the posterior
w = sample(fit, n_sample = 3)
```

    5×5×3 Array{Float32, 3}:
    [:, :, 1] =
     0.0  0.140196   0.274407  0.0   0.0
     0.0  0.0       -0.284611  0.0   0.0
     0.0  0.0        0.0       0.0  -0.281174
     0.0  0.0       -0.229761  0.0   0.190547
     0.0  0.0        0.0       0.0   0.0

    [:, :, 2] =
     0.0   0.115448   0.274675  0.0       0.0
     0.0   0.0        0.0       0.0       0.0
     0.0  -0.177611   0.0       0.0       0.0
     0.0   0.0       -0.173311  0.0       0.0
     0.0   0.0        0.0       0.228793  0.0

    [:, :, 3] =
     0.0   0.0       0.161835   0.18919   -0.291537
     0.0   0.0       0.0        0.0        0.0
     0.0   0.0       0.0       -0.16452    0.0
     0.0  -0.242377  0.0        0.0        0.0
     0.0   0.0       0.0       -0.140842   0.0

To learn a posterior over nonlinear DAGs in the form of acyclic
multilayer perceptrons (MLPs), use the `fit_mlp()` function.

``` julia
CUDA.seed!(1)
Random.seed!(1)

# Fit a posterior with a single hidden-layer of 10 neurons and a N(0,1) prior on each weight
fit = fit_mlp(x, hidden_layers = [10], bias = false, prior_μ = 0, prior_σ = 1, verbose = false)

# Draw a sample of DAGs from the posterior
w, model = sample(fit, n_sample = 3)

# The vector called "model" contains acyclic MLPs drawn from the posterior
x_new = randn(Float32, 5, p)
x̂ = model[1](x_new')'
```

    5×5 adjoint(::Matrix{Float32}) with eltype Float32:
      0.737726      0.174007  -0.149519   0.164915  0.0
     -0.197016     -0.10658    0.0728614  0.104097  0.0
      0.72916       0.122759  -0.105185   0.16743   0.0
      0.701111      0.215224  -0.166327   0.277106  0.0
     -0.000798782   0.043071  -0.227916   0.27824   0.0
