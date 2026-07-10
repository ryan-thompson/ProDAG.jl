

# ProDAG

Julia implementation of ProDAG from the NeurIPS 2025 paper [“ProDAG:
Projected variational inference for directed acyclic
graphs”](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ee42c13f231836e914930925f950fc62-Abstract-Conference.html).

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

    Precompiling packages...
       6416.5 ms  ✓ ProDAG
      1 dependency successfully precompiled in 7 seconds. 199 already precompiled.
    Precompiling packages...
       1585.7 ms  ✓ QuartoNotebookWorkerTablesExt (serial)
      1 dependency successfully precompiled in 2 seconds
    Precompiling packages...
        806.7 ms  ✓ QuartoNotebookWorkerLaTeXStringsExt (serial)
      1 dependency successfully precompiled in 1 seconds

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
      0.265896   -0.092639  1.11583   -0.0745545  0.0
     -0.371186   -0.480474  0.431186   0.0556441  0.0
      0.285064   -0.09918   1.33071   -0.0041039  0.0
      0.247853   -0.105833  1.65873    0.0199266  0.0
     -0.0514972  -0.342665  0.616304  -0.0116134  0.0
