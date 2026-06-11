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
function make_3d_function(n=1000)
    X = 10 .* rand(n,3) .- 5  # uniform in [-2,2]
    Y = [x[3] - x[1]^2 - x[2]^2 for x in eachrow(X)]
    return X, Y
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
X, Y = make_3d_function(10000)
Y = reshape(Y, :, 1)  # shape (m,1)

# Convert to Flux-friendly shapes (features × batch)
X_tensor = permutedims(X)
Y_tensor = permutedims(Y)

# ----------------------------
# Build model & optimizer
# ----------------------------
n = [3, 4, 1]
model = build_network(n, normalization = "layernorm")
loss_fn(m, x, y) = mse(m(x), y)
opt = ADAM(0.01)
opt_state = Flux.setup(opt, model)

# ----------------------------
# Training loop
# ----------------------------
epochs_per_frame = 10
total_frames = 50
data = [(X_tensor, Y_tensor)]  # single batch
model_snapshots = Flux.Chain[]
for frame in 0:total_frames
    for epoch in 1:epochs_per_frame
        Flux.train!(loss_fn, model, data, opt_state)
    end
    println("Frame $frame, Loss: ", loss_fn(model, X_tensor, Y_tensor))

    push!(model_snapshots, deepcopy(model))
end
CAB_analysis.plot_empirical_CAB_3D(model_snapshots[end], 2, 1, total_frames)

