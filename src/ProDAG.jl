# =================================================================================================#
# Description: Implementation of ProDAG
# Author: Ryan Thompson
# =================================================================================================#

module ProDAG

import CUDA, Flux, Graphs, LinearAlgebra, Optimisers, Printf, Zygote

export fit_linear, fit_mlp, sample

#==================================================================================================#
# Function that reimplements Flux.early_stopping with <= instead of <
#==================================================================================================#

function early_stopping(f, delay; distance = -, init_score = 0, min_dist = 0)
    trigger = let best_score = init_score
      (args...; kwargs...) -> begin
        score = f(args...; kwargs...)
        Δ = distance(best_score, score)
        best_score = Δ < 0 ? best_score : score
        return Δ <= min_dist
      end
    end
    return Flux.patience(trigger, delay)
  end

#==================================================================================================#
# Functions that perform batched matrix computations
#==================================================================================================#

# Adapted from https://github.com/JuliaGPU/CUDA.jl/blob/master/lib/cublas/wrappers.jl
for (fname1, fname2, elty) in
    ((:cublasDgetrfBatched, :cublasDgetriBatched, :Float64),
     (:cublasSgetrfBatched, :cublasSgetriBatched, :Float32),
     (:cublasZgetrfBatched, :cublasZgetriBatched, :ComplexF64),
     (:cublasCgetrfBatched, :cublasCgetriBatched, :ComplexF32))
    @eval begin
        function matinv_batched!(A::Vector{<:CUDA.StridedCuMatrix{$elty}}, 
            C::Vector{<:CUDA.StridedCuMatrix{$elty}})
            batchSize = length(A)
            n = size(A[1], 1)
            lda = max(1, stride(A[1], 2))
            ldc = max(1, stride(C[1], 2))
            Aptrs = CUDA.CUBLAS.unsafe_batch(A)
            Cptrs = CUDA.CUBLAS.unsafe_batch(C)
            info = CUDA.zeros(Cint, batchSize)
            CUDA.CUBLAS.$fname1(CUDA.CUBLAS.handle(), n, Aptrs, lda, CUDA.CU_NULL, info, batchSize)
            CUDA.CUBLAS.$fname2(CUDA.CUBLAS.handle(), n, Aptrs, lda, CUDA.CU_NULL, Cptrs, ldc, info,
                batchSize)
            CUDA.CUBLAS.unsafe_free!(Aptrs)
            CUDA.CUBLAS.unsafe_free!(Cptrs)
        end
    end
end

function xw_mult(w; x = x)
    x_view = view(reshape(x, size(x, 1), size(x, 2), 1), :, :, :)
    CUDA.CUBLAS.gemm_strided_batched('N', 'N', 1, x_view, w)
end

function Zygote.rrule(::typeof(xw_mult), w; x = x)
    y = xw_mult(w; x = x)
    function y_pullback(dy)
        x_view = view(reshape(x, size(x, 1), size(x, 2), 1), :, :, :)
        dw = CUDA.CUBLAS.gemm_strided_batched('T', 'N', 1.0, x_view, dy)
        (Zygote.NoTangent(), dw)
    end
    y, y_pullback
end

#==================================================================================================#
# Function to project weighted adjancency matrix onto set of DAGs
#==================================================================================================#

function project_dag(w̃, params)

    # Save params
    s, mu, alpha, tol, max_step, max_iter, threshold, lr = params

    # Compute scaling constant
    max_ = maximum(abs.(w̃), dims = (1, 2))
    max_ = ifelse.(max_ .> 0.0f0, max_, 1.0f0)

    # Initialise variables
    w = zero(w̃)
    I = zero(w̃) .+ LinearAlgebra.Diagonal(one(w̃[:, :, 1]))

    # matinv_batched! does in-place inverse
    sIw2 = similar(w̃)
    sIw2_inv = similar(w̃)

    # matinv_batched! expects a vector of matrices rather than a 3d array
    sIw2_batch = collect(eachslice(sIw2, dims = 3))
    sIw2_inv_batch = collect(eachslice(sIw2_inv, dims = 3))

    # Perform DAG projection
    for step in 1:max_step

        # Run gradient descent
        for iter in 1:max_iter

            # Compute gradients
            sIw2 .= permutedims(s * I - w .^ 2, (2, 1, 3))
            matinv_batched!(sIw2_batch, sIw2_inv_batch)
            grad = 2 * sIw2_inv .* w + mu * (w - w̃ ./ max_)

            # Take gradient descent step
            w .-= lr .* grad

            # Check for convergence
            if maximum(abs.(grad)) <= tol
                break
            end

        end

        # Update mu
        mu *= alpha

    end

    # Rescale weights
    w .*= max_

    # Threshold small weights
    w[abs.(w) .<= threshold] .= 0
    w[w .≠ 0] = w̃[w .≠ 0]

    w

end

#==================================================================================================#
# Function to project weighted adjancency matrix onto ℓ1-ball
#==================================================================================================#

function project_l1(w, λ)

    # Save input dims
    dims = size(w)

    # If already in ℓ1 ball or λ=0 then exit early
    if all(vec(sum(abs.(w), dims = (1, 2))) .<= λ)
        return w
    elseif iszero(λ)
        return zero(w)
    end

    # Flatten to vector
    w = reshape(w, dims[1] * dims[2], dims[3])

    # Run algorithm for projection onto simplex by Duchi et al. (2008, ICML)
    w_abs = abs.(w)
    w_sort = sort(w_abs, rev = true, dims = 1)
    csum = cumsum(w_sort, dims = 1)
    indices = Flux.gpu(collect(1:dims[1] * dims[2]))
    max_j = vec(maximum((w_sort .* indices .> csum .- reshape(λ, 1, dims[3])) .* indices, dims = 1))
    theta = max.((csum[CartesianIndex.(max_j, 1:dims[3])] .- λ) ./ max_j, 0.0f0)
    w_abs = max.(w_abs .- transpose(theta), 0.0f0)
    w = w_abs .* sign.(w)

    # Return to original shape
    reshape(w, dims)

end

#==================================================================================================#
# Function to project weighted adjancency matrix onto intersection of sets
#==================================================================================================#

function project(w̃, λ; params)

    # Project onto DAG set
    w = project_dag(w̃, params)

    # Project onto ℓ1-ball
    w = project_l1(w, λ)

    w

end

function Zygote.rrule(::typeof(project), w̃, λ; params)

    # Compute optimal solution
    w = project(w̃, λ, params = params)

    # Configure vector-Jacobian product (vJp)
    function w_pullback(dw)
        if iszero(λ)
            dw̃ = zero(w)
            dλ = zero(λ)
        else
            A = w .≠ 0
            dλ = sum(sign.(w) .* dw, dims = (1, 2)) ./ sum(A, dims = (1, 2))
            dw̃ = A .* dw .- dλ .* sign.(w)
            non_binding = .!isapprox.(vec(sum(abs.(w), dims = (1, 2))), λ)
            if any(non_binding)
                dw̃[:, :, non_binding] .= A[:, :, non_binding] .* dw[:, :, non_binding]
                dλ[:, :, non_binding] .= 0
            end
            dλ = reshape(dλ, :)
        end
        (Zygote.NoTangent(), dw̃, dλ)
    end

    w, w_pullback

end

#==================================================================================================#
# Functions for KL divergences
#==================================================================================================#
    
# KL divergence between normal distributions
function kl_norm(μ_q, σ_q, μ_p, σ_p)
    log(σ_p / σ_q) + (σ_q ^ 2 + (μ_q - μ_p) ^ 2) / (2 * σ_p ^ 2) - 0.5
end

# KL divergence between exponential distributions (parameterized by mean α)
function kl_exp(α_q, α_p)
    log(α_p / α_q) + α_q / α_p - 1
end

#==================================================================================================#
# Function to train model
#==================================================================================================#

# So CUDA.ones(), CUDA.rand(), and CUDA.randn() have the gradients of ones(), rand(), and randn()
Zygote.@adjoint CUDA.ones(args...) = CUDA.ones(args...), Δ -> (nothing,)
Zygote.@adjoint CUDA.rand(args...) = CUDA.rand(args...), Δ -> (nothing,)
Zygote.@adjoint CUDA.randn(args...) = CUDA.randn(args...), Δ -> (nothing,)

# Function to train linear model
function train_linear!(x, μ, cont_σ, cont_α, prior_μ, prior_σ, prior_α, params, epoch_max, patience, 
    optimiser, optimiser_args, n_sample, dirac_λ, verbose)

    # Instantiate optimiser and collect variational parameters
    optim = optimiser(optimiser_args...)
    vp = Flux.params(μ, cont_σ, cont_α)

    # Set convergence criterion
    converge = early_stopping(x -> x, patience, init_score = Inf)

    # Save data dimensions
    n, p = size(x)

    # Create objective function (ELBO)
    function objective(μ, cont_σ, cont_α)

        # Rescale variational parameters
        σ = Flux.softplus(cont_σ)
        # σ = abs.(cont_σ)

        # Compute KL divergence
        kl = sum(kl_norm.(μ, σ, prior_μ, prior_σ))

        # Sample sparsity parameters
        if dirac_λ
            λ = prior_α .* CUDA.ones(n_sample)
        else
            α = Flux.softplus(cont_α)
            λ = - α .* log.(CUDA.rand(n_sample))
            kl += sum(kl_exp.(α, prior_α))
        end

        # Sample weighted adjacency matrices
        μ = reshape(μ, p, p)
        σ = reshape(σ, p, p)
        ε = CUDA.randn(size(μ)..., n_sample)
        w̃ = μ .+ σ .* ε
        w = project(w̃, λ, params = params)

        # Compute expected log-likelihood
        x̂ = xw_mult(w; x = x)
        ell = - 0.5 * sum((x̂ .- x) .^ 2) / n_sample - 0.5 * n * p * log(2 * pi)

        # Compute negative ELBO
        (- ell + kl) / n

    end
    
    # Run optimisation
    for epoch in 1:epoch_max

        # Record negative ELBO and gradients
        neg_elbo, grad = Flux.withgradient(() -> objective(μ, cont_σ, cont_α), vp)

        # Check for convergence
        converge(neg_elbo) && break

        # Update variational parameters
        Flux.update!(optim, vp, grad)

        # Print status update
        if verbose
            # Printf.@printf("\33[2K\rEpoch: %i, Neg. ELBO: %.4f", epoch, neg_elbo)
            Printf.@printf("Epoch: %i, Neg. ELBO: %.4f \n", epoch, neg_elbo)
        end

    end

end

# Function to train multilayer perceptron model
function train_mlp!(x, construct, μ, cont_σ, cont_α, prior_μ, prior_σ, prior_α, params, epoch_max, 
    patience, optimiser, optimiser_args, n_sample, dirac_λ, verbose, output_ind, input_ind, 
    ind_order, ind_mat)

    # Instantiate optimiser and collect variational parameters
    optim = optimiser(optimiser_args...)
    vp = Flux.params(μ, cont_σ, cont_α)

    # Set convergence criterion
    converge = early_stopping(x -> x, patience, init_score = Inf)

    # Save data dimensions
    p, n = size(x)

    # Create objective function (ELBO)
    function objective(μ, cont_σ, cont_α)

        # Rescale variational parameters
        σ = Flux.softplus(cont_σ)

        # Compute KL divergence
        kl = sum(kl_norm.(μ, σ, prior_μ, prior_σ))

        # Sample sparsity parameters
        if dirac_λ
            λ = prior_α .* CUDA.ones(n_sample)
        else
            α = Flux.softplus(cont_α)
            λ = - α .* log.(CUDA.rand(n_sample))
            kl += sum(kl_exp.(α, prior_α))
        end

        # Sample weighted adjacency matrices
        ε = CUDA.randn(size(μ)..., n_sample)
        ῶ = μ .+ σ .* ε
        ῶ_hidden = ῶ[input_ind, :]
        ω_output = ῶ[output_ind, :]
        w̃ = reshape(sqrt.(ind_mat * ῶ_hidden .^ 2), p, p, n_sample)
        w = project(w̃, λ, params = params)
        scale_factor = transpose(ind_mat) * reshape(w ./ w̃, p * p, n_sample)
        ω_hidden = ῶ_hidden .* scale_factor
        ω = vcat(ω_hidden, ω_output)[ind_order, :]

        # Compute expected log-likelihood
        ell = 0
        for i in 1:n_sample
            model_i = construct(ω[:, i])
            x̂ = model_i(x)
            ell -= 0.5 * sum((x̂ .- x) .^ 2) / n_sample 
        end
        ell -= 0.5 * n * p * log(2 * pi)

        # Compute negative ELBO
        (- ell + kl) / n

    end

    # Run optimisation
    for epoch in 1:epoch_max

        # Record negative ELBO and gradients
        neg_elbo, grad = Flux.withgradient(() -> objective(μ, cont_σ, cont_α), vp)

        # Check for convergence
        converge(neg_elbo) && break

        # Update variational parameters
        Flux.update!(optim, vp, grad)

        # Print status update
        if verbose
            # Printf.@printf("\33[2K\rEpoch: %i, Neg. ELBO: %.4f", epoch, neg_elbo)
            Printf.@printf("Epoch: %i, Neg. ELBO: %.4f \n", epoch, neg_elbo)
        end

    end

end

#==================================================================================================#
# Functions to create multilayer perceptron network
#==================================================================================================#

function create_mlp(n_neuron, p, activation_fun)

    # Create a neural network for each variable
    subnetwork = Vector{Flux.Chain}(undef, p)
    for j in 1:p
        subnetwork[j] = Flux.Chain(
            Flux.Dense(p, n_neuron, activation_fun, bias = false), 
            Flux.Dense(n_neuron, 1, bias = false)
            )
    end

    # Concatenate individual neural networks
    model = Flux.Parallel(vcat, subnetwork...)

    # Destructure the network so we can construct it later using sampled parameters
    _, construct = Flux.destructure(model)

    # Create an indicator matrix
    ind = repeat(1:p ^ 2, inner = n_neuron)
    ind_mat = zeros(p ^ 2, p ^ 2 * n_neuron)
    for j in 1:p ^ 2 * n_neuron
        ind_mat[ind[j], j] = 1
    end

    # Create indexes
    n_hidden = p * n_neuron
    n_total = p * n_neuron + n_neuron
    output_ind = vcat([n_hidden + 1 + (j - 1) * n_total:n_total + (j - 1) * n_total for j in 1:p]...)
    input_ind = setdiff(1:p ^ 2 * n_neuron + p * n_neuron, output_ind)
    ind_order = sortperm(vcat(input_ind, output_ind))

    construct, output_ind, input_ind, ind_order, ind_mat

end

#==================================================================================================#
# Type for model
#==================================================================================================#

# Type for linear model
struct ProDAGLinearFit
    μ::Vector{<:Real} # Posterior means of w̃
    σ::Vector{<:Real} # Posterior standard deviations of w̃
    α::Vector{<:Real} # Posterior mean of λ
    dirac_λ::Bool # Prior on λ is Dirac or exponential
    p::Int # Number of nodes
end

# Type for multilayer perceptron model
struct ProDAGMLPFit
    μ::Vector{<:Real} # Posterior means of w̃
    σ::Vector{<:Real} # Posterior standard deviations of w̃
    α::Vector{<:Real} # Posterior mean of λ
    dirac_λ::Bool # Prior on λ is Dirac or exponential
    p::Int # Number of nodes
    construct::Optimisers.Restructure # Function to construct neural network from weights
    output_ind::Vector{<:Real} # Indexes of the output layer weights
    input_ind::Vector{<:Real} # Indexes of the input layer weights
    ind_order::Vector{<:Real} # Inverts the above
    ind_mat::Matrix{<:Real} # For computing the 2-norms of the input layer weights
end

#==================================================================================================#
# Function to fit model
#==================================================================================================#

# Function to fit linear model
"""
fit_linear(x; <keyword arguments>)

Performs a Bayesian fit of a linear  DAG to variables `x` using projection-induced distributions.

# Arguments
- `prior_μ = 0.0`: the prior mean of w̃; can be a scalar or a `size(x, 2) ^ 2` vector.
- `prior_σ = 1.0`: the prior standard deviation of w̃; can be a scalar or a `size(x, 2) ^ 2` vector.
- `prior_α = Inf`: the prior mean of λ.
- `init_μ = prior_μ`: the initial value of the posterior mean of w̃; can be a scalar or a \
`size(x, 2) ^ 2` vector.
- `init_σ = prior_σ`: the initial value of the posterior standard deviation of w̃; can be a scalar \
or a `size(x, 2) ^ 2` vector.
- `init_α = prior_α`: the initial value of the posterior mean of λ; can be a scalar or a \
`size(x, 2) ^ 2` vector.
- `dirac_λ = true`: if `true`, λ is modeled as a Dirac delta distribution centred on `prior_α`, \
otherwise if `false`, λ is modeled as an exponential distribution with mean `prior_α`
- `epoch_max = 1000`: the maximum number of training epochs.
- `patience = 5`: the number of epochs to wait before declaring convergence.
- `optimiser = Flux.Adam`: an optimiser from Flux to use for training.
- `optimiser_args = (0.1)`: a tuple of arguments to pass to `optimiser`.
- `params = (1, 1, 0.5, 1e-2, 10, 10000, 0.1, 1 / size(x, 2))`: parameters for the acyclicity \
projection in the following order: log det parameter `s`, path coefficient `μ`, decay factor `c`, \
convergence tolerance `tol`, step count `T`, maximum gradient descent iterations `max_iter`, \
thresholding parameter `threshold`, learning rate `lr`.
- `n_sample = 1000`: the number of samples of `w` to draw when estimating the objective function.
- `verbose = true`: whether to print status updates during training.
``

See also [`sample`](@ref).
"""
function fit_linear(x; prior_μ = 0.0, prior_σ = 1.0, prior_α = Inf, init_μ = prior_μ, 
    init_σ = prior_σ, init_α = prior_α, dirac_λ = true, epoch_max = 1000, patience = 5, 
    optimiser = Flux.Adam, optimiser_args = (0.1), 
    params = (1, 1, 0.5, 1e-2, 10, 10000, 0.1, 1 / size(x, 2)), n_sample = 1000, verbose = true)

    # Save data dimension
    p = size(x, 2)

    # Save number of parameters
    n_weights = p ^ 2

    # Create priors
    prior_μ = prior_μ .* ones(n_weights)
    prior_σ = prior_σ .* ones(n_weights)
    prior_α = prior_α .* ones(1)

    # Initialize variational parameters
    μ = deepcopy(init_μ) .* ones(n_weights)
    σ = deepcopy(init_σ) .* ones(n_weights)
    α = deepcopy(init_α) .* ones(1)
    cont_σ = log.(exp.(σ) .- 1)
    # cont_σ = σ
    cont_α = log.(exp.(α) .- 1)

    # Move data and parameters to GPU
    x = Flux.gpu(x)
    prior_μ = Flux.gpu(prior_μ)
    prior_σ = Flux.gpu(prior_σ)
    prior_α = Flux.gpu(prior_α)
    μ = Flux.gpu(μ)
    cont_σ = Flux.gpu(cont_σ)
    cont_α = Flux.gpu(cont_α)

    # Train the variational posterior
    train_linear!(x, μ, cont_σ, cont_α, prior_μ, prior_σ, prior_α, params, epoch_max, patience, 
        optimiser, optimiser_args, n_sample, dirac_λ, verbose)

    # Move parameters to CPU
    μ = Flux.cpu(μ)
    cont_σ = Flux.cpu(cont_σ)
    cont_α = Flux.cpu(cont_α)
    σ = Flux.softplus(cont_σ)
    α = Flux.softplus(cont_α)

    ProDAGLinearFit(μ, σ, α, dirac_λ, p)

end

# Function to fit multilayer perceptron model
"""
fit_mlp(x; <keyword arguments>)

Performs a Bayesian fit of a nonlinear (multilayer perceptron) DAG to variables `x` using \
projection-induced distributions.

# Arguments
- `n_neuron = 10`: the number of neurons to use in the hidden layer of the MLP.
- `activation_fun = Flux.relu`: the activation function to use in the hidden layer of the MLP.
- `prior_μ = 0.0`: the prior mean of w̃; can be a scalar or a `size(x, 2) ^ 2` vector.
- `prior_σ = 1.0`: the prior standard deviation of w̃; can be a scalar or a `size(x, 2) ^ 2` vector.
- `prior_α = Inf`: the prior mean of λ.
- `init_μ = prior_μ`: the initial value of the posterior mean of w̃; can be a scalar or a \
`size(x, 2) ^ 2` vector.
- `init_σ = prior_σ`: the initial value of the posterior standard deviation of w̃; can be a scalar \
or a `size(x, 2) ^ 2` vector.
- `init_α = prior_α`: the initial value of the posterior mean of λ; can be a scalar or a \
`size(x, 2) ^ 2` vector.
- `dirac_λ = true`: if `true`, λ is modeled as a Dirac delta distribution centred on `prior_α`, \
otherwise if `false`, λ is modeled as an exponential distribution with mean `prior_α`
- `epoch_max = 1000`: the maximum number of training epochs.
- `patience = 5`: the number of epochs to wait before declaring convergence.
- `optimiser = Flux.Adam`: an optimiser from Flux to use for training.
- `optimiser_args = (0.1)`: a tuple of arguments to pass to `optimiser`.
- `params = (1, 1, 0.5, 1e-2, 10, 10000, 0.1, 0.25 / size(x, 2))`: parameters for the acyclicity \
projection in the following order: log det parameter `s`, path coefficient `μ`, decay factor `c`, \
convergence tolerance `tol`, step count `T`, maximum gradient descent iterations `max_iter`, \
thresholding parameter `threshold`, learning rate `lr`.
- `n_sample = 1000`: the number of samples of `w` to draw when estimating the objective function.
- `verbose = true`: whether to print status updates during training.
``

See also [`sample`](@ref).
"""
function fit_mlp(x; n_neuron = 10, activation_fun = Flux.relu, prior_μ = 0.0, 
    prior_σ = 1, prior_α = Inf, init_μ = prior_μ, init_σ = prior_σ, init_α = prior_α, 
    dirac_λ = true, epoch_max = 1000, patience = 5, optimiser = Flux.Adam, 
    optimiser_args = (0.1), params = (1, 1, 0.5, 1e-2, 10, 10000, 0.1, 0.25 / size(x, 2)), 
    n_sample = 1000, verbose = true)

    # Save data dimension
    p = size(x, 2)

    # Construct neural network
    construct, output_ind, input_ind, ind_order, ind_mat = create_mlp(n_neuron, p, activation_fun)

    # Save number of parameters
    n_weights = p ^ 2 * n_neuron + p * n_neuron

    # Create priors
    prior_μ = prior_μ .* ones(n_weights)
    prior_σ = prior_σ .* ones(n_weights)
    prior_α = prior_α .* ones(1)

    # Initialize variational parameters
    μ = deepcopy(init_μ) .* ones(n_weights)
    σ = deepcopy(init_σ) .* ones(n_weights)
    α = deepcopy(init_α) .* ones(1)
    cont_σ = log.(exp.(σ) .- 1)
    cont_α = log.(exp.(α) .- 1)

    # Move data and parameters to GPU
    x = transpose(x)
    x = Flux.gpu(x)
    prior_μ = Flux.gpu(prior_μ)
    prior_σ = Flux.gpu(prior_σ)
    prior_α = Flux.gpu(prior_α)
    μ = Flux.gpu(μ)
    cont_σ = Flux.gpu(cont_σ)
    cont_α = Flux.gpu(cont_α)
    ind_mat = Flux.gpu(ind_mat)

    # Train the variational posterior
    train_mlp!(x, construct, μ, cont_σ, cont_α, prior_μ, prior_σ, prior_α, params, epoch_max, 
        patience, optimiser, optimiser_args, n_sample, dirac_λ, verbose, output_ind, input_ind, 
        ind_order, ind_mat)

    # Move parameters to CPU
    μ = Flux.cpu(μ)
    cont_σ = Flux.cpu(cont_σ)
    cont_α = Flux.cpu(cont_α)
    σ = Flux.softplus(cont_σ)
    α = Flux.softplus(cont_α)
    ind_mat = Flux.cpu(ind_mat)

    ProDAGMLPFit(μ, σ, α, dirac_λ, p, construct, output_ind, input_ind, ind_order, ind_mat)

end

#==================================================================================================#
# Function to sample weighted adjacency matrices from fitted model
#==================================================================================================#

# Function that deletes smallest edges until w is a DAG
function guarantee_dag!(w)
    for i in 1:size(w, 3)
        while Graphs.is_cyclic(Graphs.SimpleDiGraph(w[:, :, i]))
            nonzero_ind = findall(!iszero, w[:, :, i])
            if isempty(nonzero_ind)
                break
            end
            nonzero_value = [w[ind[1], ind[2], i] for ind in nonzero_ind]
            min_ind = argmin(abs.(nonzero_value))
            w[nonzero_ind[min_ind][1], nonzero_ind[min_ind][2], i] = 0.0
        end
    end
end

# Function to sample from fitted linear model
"""
fit_mlp(fit; <keyword arguments>)

Samples DAGs from a fitted Bayesian posterior distribution.

# Arguments
- `n_sample = 1000`: the number of samples of `w` to draw when estimating the objective function.
- `gurantee_dag = true`: whether to threshold the adjacency matrix to guarantee that all cycles \
are removed.
"""
function sample(fit::ProDAGLinearFit; n_sample = 1000, guarantee_dag = true, 
    params = (1, 1, 0.5, 1e-4, 10, 10000, 0.1, 1 / fit.p))
    
    # Move variational parameters to GPU
    μ = Flux.gpu(fit.μ)
    σ = Flux.gpu(fit.σ)
    α = Flux.gpu(fit.α)

    # Sample sparsity parameters
    if fit.dirac_λ
        λ = α .* CUDA.ones(n_sample)
    else
        λ = - α .* log.(CUDA.rand(n_sample))
    end

    # Sample weighted adjacency matrices
    μ = reshape(μ, fit.p, fit.p)
    σ = reshape(σ, fit.p, fit.p)
    ε = CUDA.randn(size(μ)..., n_sample)
    w̃ = μ .+ σ .* ε
    w = project(w̃, λ, params = params)

    # Move weighted adjacency matrices to CPU
    w = Flux.cpu(w)

    # Guarantee output is a DAG
    if guarantee_dag
        guarantee_dag!(w)
    end

    w

end

# Function to sample from fitted multilayer perceptron model
function sample(fit::ProDAGMLPFit; n_sample = 1000, guarantee_dag = true, 
    params = (1, 1, 0.5, 1e-4, 10, 10000, 0.1, 0.25 / fit.p))

    # Move variational parameters to GPU
    μ = Flux.gpu(fit.μ)
    σ = Flux.gpu(fit.σ)
    α = Flux.gpu(fit.α)
    ind_mat = Flux.gpu(fit.ind_mat)

    # Sample sparsity parameters
    if fit.dirac_λ
        λ = α .* CUDA.ones(n_sample)
    else
        λ = - α .* log.(CUDA.rand(n_sample))
    end

    # Sample weighted adjacency matrices
    ε = CUDA.randn(size(μ)..., n_sample)
    ῶ = μ .+ σ .* ε
    ῶ_hidden = ῶ[fit.input_ind, :]
    ω_output = ῶ[fit.output_ind, :]
    w̃ = reshape(sqrt.(ind_mat * ῶ_hidden .^ 2), fit.p, fit.p, n_sample)
    w = project(w̃, λ, params = params)

    # Guarantee output is a DAG
    if guarantee_dag
        w = Flux.cpu(w)
        guarantee_dag!(w)
        w = Flux.gpu(w)
    end

    scale_factor = transpose(ind_mat) * reshape(w ./ w̃, fit.p * fit.p, n_sample)
    ω_hidden = ῶ_hidden .* scale_factor
    ω = vcat(ω_hidden, ω_output)[fit.ind_order, :]

    # Move weighted adjacency matrices to CPU
    w = Flux.cpu(w)
    ω = Flux.cpu(ω)

    # Construct neural networks from weights
    model = Vector{Flux.Parallel}(undef, n_sample)
    for i in 1:n_sample
        model[i] = fit.construct(ω[:, i])
    end

    w, model

end

end