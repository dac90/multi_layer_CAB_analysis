using Flux
using Flux.Losses
using Statistics
using Random
using Serialization
using GLMakie
using Printf
using LinearAlgebra
using Distributions
using Revise
using CAB_analysis  # local package

# ----------------------------
# Network architecture & data
# ----------------------------
n = [2, 6, 1]
m = 1000

# ----------------------------
# Target function
# ----------------------------
target_function(x, y) = x .* y .- 1

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
# Generate training data
# ----------------------------
x_vals = rand(Uniform(-5, 5), m, 1)
y_vals = rand(Uniform(-5, 5), m, 1)
X = hcat(x_vals, y_vals)
Y = target_function(x_vals, y_vals)

# Convert to Flux-friendly shapes (features × batch)
X_tensor = permutedims(X)  # 2 × m
Y_tensor = permutedims(Y)  # 1 × m

# ----------------------------
# Build model & optimizer
# ----------------------------
model = build_network(n, normalization = "layernorm")
loss_fn(m, x, y) = mse(m(x), y)
opt = RMSProp()
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

    # CAB analysis
    #CAB_analysis.calculate_CAB_partition_tree(model, n, 1, frame)
    #CAB_analysis.calculate_CAB_neuron_table(model, n, frame)
end

# ----------------------------
# CAB_analysis visualization
# ----------------------------
#CAB_analysis.get_CAB_partition_tree(100)
#CAB_analysis.plot_CAB(n, 3, 1, 100)
CAB_analysis.plot_empirical_CAB(model, 3, 1, 20)

#CAB_analysis.create_animation(n, total_frames)
#CAB_analysis.create_animation_2(n, total_frames)
CAB_analysis.create_empirical_animation(model_snapshots, total_frames)
#CAB_analysis.plot_partition_count(total_frames)
