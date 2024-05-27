

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
using ProDAG, Random

Random.seed!(1)

# Generate some data
n = 100
p = 5
x = randn(n, p)

# Fit a posterior over linear DAGs with a N(0,1) prior on each weight
fit = fit_linear(x, prior_μ = 0, prior_σ = 1, verbose = false)

# Draw a sample of DAGs from the posterior
w = sample(fit, n_sample = 3)
```

    5×5×3 Array{Float32, 3}:
    [:, :, 1] =
     0.0   0.0        0.189677  0.12658  -0.185433
     0.0   0.0        0.0       0.0       0.0
     0.0   0.0        0.0       0.0       0.0
     0.0  -0.148973   0.142132  0.0       0.0
     0.0   0.219555  -0.167325  0.0       0.0

    [:, :, 2] =
     0.0  0.24542   0.0        0.0       -0.209093
     0.0  0.0       0.0       -0.310528   0.0
     0.0  0.159327  0.0        0.0        0.0
     0.0  0.0       0.0        0.0        0.0
     0.0  0.0       0.268417   0.0        0.0

    [:, :, 3] =
     0.0      0.0       0.0   0.0       0.0
     0.0      0.0       0.0   0.0       0.204187
     0.19038  0.303957  0.0   0.0       0.0
     0.0      0.0       0.0   0.0       0.0
     0.0      0.0       0.0  -0.181489  0.0

To learn a posterior over nonlinear DAGs in the form of acyclic
multilayer perceptrons (MLPs), use the `fit_mlp()` function.

``` julia
# Fit a posterior over nonlinear DAGs with a N(0,1) prior on each weight
fit = fit_mlp(x, prior_μ = 0, prior_σ = 1, verbose = false)

# Draw a sample of DAGs from the posterior
w, model = sample(fit, n_sample = 3)

# The vector called "model" contains acyclic MLPs drawn from the posterior
x_new = randn(Float32, 5, p)
x̂ = model[1](x_new')'
```

    5×5 adjoint(::Matrix{Float32}) with eltype Float32:
     -0.0698817  0.429869   0.0760589  -0.147528   0.0
      0.701148   0.244819   1.01625     0.239965   0.0
      0.166468   0.0773325  0.0236804   0.135913   0.0
      0.513051   0.319638   0.927851    0.183975   0.0
      0.0135465  0.300484   0.130642   -0.0806653  0.0
