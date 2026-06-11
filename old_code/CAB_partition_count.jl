using Flux
using Flux.Losses
using Statistics
using Random
using Serialization
using GLMakie
using Printf
using LinearAlgebra
using Revise
using CAB_analysis  # local package
# ----------------------------
# Helper: approximate moons dataset
# ----------------------------
function make_moons(n_samples=1000; noise=0.1)
    n_samples_out = n_samples ÷ 2
    n_samples_in = n_samples - n_samples_out

    θ1 = range(0, π, length=n_samples_out)
    x1 = hcat(cos.(θ1), sin.(θ1))          # shape: (n_samples_out, 2)

    θ2 = range(0, π, length=n_samples_in)
    x2 = hcat(1 .- cos.(θ2), 0 .- sin.(θ2) .+ 0.5)  # shape: (n_samples_in, 2)

    X = vcat(x1, x2) .* 2.0
    y = vcat(zeros(n_samples_out), ones(n_samples_in))

    if noise > 0
        X .+= noise .* randn(size(X))
    end

    return X, y
end

# ----------------------------
# Flexible ReLU network
# ----------------------------
function build_network(layer_sizes::Vector{Int}; normalization::Union{Nothing,String}=nothing)
    layers = []

    for i in 2:length(layer_sizes)
        in_dim = layer_sizes[i-1]
        out_dim = layer_sizes[i]

        if i == length(layer_sizes)
            push!(layers, Dense(in_dim, out_dim, identity))
        elseif i == length(layer_sizes) - 1 && normalization=="layernorm"
            push!(layers, Dense(in_dim, out_dim, identity))
            push!(layers, LayerNorm(out_dim, relu))
        else
            push!(layers, Dense(in_dim, out_dim, relu))
        end
    end

    return Flux.Chain(layers...)
end

# ----------------------------
# Generate dataset
# ----------------------------
X, Y = make_moons(1000, noise=0.1)
Y = reshape(Y, :, 1)  # shape (m,1)

# Convert to Flux-friendly shapes (features × batch)
X_tensor = permutedims(X)
Y_tensor = permutedims(Y)
data = [(X_tensor, Y_tensor)]  # single batch

# ----------------------------
# Build model & optimizer
# ----------------------------
total_cycles = 100
epochs_per_frame = 10
total_frames = 50
n = [2, 4, 4, 1]
loss_fn(m, x, y) = logitbinarycrossentropy(m(x), y)

boundary_count = Array{Int}(undef, total_cycles, total_frames)
null_count = Array{Int}(undef, total_cycles, total_frames)
non_boundary_count = Array{Int}(undef, total_cycles, total_frames)
loss = Array{Float64}(undef, total_cycles, total_frames)

# ----------------------------
# Training loop
# ----------------------------
for cycle in 1:total_cycles
    println("=== Starting cycle $cycle ===")

    # reinitialize model + optimizer at start of each cycle
    model     = build_network(n)
    opt       = ADAM(0.01)
    opt_state = Flux.setup(opt, model)

    for frame in 1:total_frames
        # train for some epochs
        for epoch in 1:epochs_per_frame
            Flux.train!(loss_fn, model, data, opt_state)
        end

        # log loss
        println("Cycle $cycle, Frame $frame, Loss: ", loss_fn(model, X_tensor, Y_tensor))

        # CAB analysis
        CAB_analysis.calculate_CAB_partition_tree(model, n, 1, frame)
        CAB_analysis.calculate_CAB_neuron_table(model, n, frame)

        boundary_count[cycle,frame], null_count[cycle,frame], non_boundary_count[cycle,frame] = CAB_analysis.get_partition_count(model, n, 1)
        loss[cycle,frame] = loss_fn(model, X_tensor, Y_tensor)
    end
end
CAB_analysis.plot_partition_count(boundary_count, null_count, non_boundary_count, loss)
CAB_analysis.plot_partition_count_quantiles(boundary_count, null_count, non_boundary_count, loss)



