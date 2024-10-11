

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
     0.0        0.0       0.0       0.0  0.0
     0.277632   0.0       0.0       0.0  0.0
     0.0       -0.226752  0.0       0.0  0.0
     0.0        0.0       0.0       0.0  0.0
     0.0        0.0       0.256035  0.0  0.0

    [:, :, 2] =
      0.0       0.0  0.125841   0.0       0.0
      0.0       0.0  0.0        0.0       0.0
      0.0       0.0  0.0        0.0       0.0
     -0.179269  0.0  0.152216   0.0       0.0
      0.0       0.0  0.192518  -0.281626  0.0

    [:, :, 3] =
      0.0        0.0       0.0  0.0   0.0
      0.0        0.0       0.0  0.0   0.201371
     -0.178124   0.0       0.0  0.0   0.14015
     -0.111101  -0.161704  0.0  0.0  -0.122903
      0.0        0.0       0.0  0.0   0.0

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
      0.0634377  -0.00308696  0.23294   -0.0675012  0.0
     -0.36269    -0.160028    0.673798   0.0503829  0.0
      0.191729   -0.0644585   0.423387  -0.0134916  0.0
      0.27577    -0.104044    0.685354  -0.0163412  0.0
     -0.409075   -0.346833    0.573067   0.0475403  0.0
