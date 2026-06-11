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
        elseif i == length(layer_sizes) - 1 && normalization=="batchnorm"
            push!(layers, Dense(in_dim, out_dim, identity))
            push!(layers, BatchNorm(out_dim, relu))
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

# ----------------------------
# Ensure plot directory exists
# ----------------------------
isdir("plot_store") || mkpath("plot_store")

# ----------------------------
# Plot moons dataset
# ----------------------------
f = Figure(size=(500,500))
ax = Axis(f[1,1])
scatter!(ax, X[:,1], X[:,2], color=Y[:,1], colormap=:Spectral)
ax.xlabel = "x₁"
ax.ylabel = "x₂"
ax.title = "Moons Dataset"
save("plot_store/moons_plot.png", f)

# ----------------------------
# Build model & optimizer
# ----------------------------
n = [2, 3, 3, 1]
model = build_network(n, normalization = "layernorm")
loss_fn(m, x, y) = logitbinarycrossentropy(m(x), y)
opt = ADAM(0.01)
opt_state = Flux.setup(opt, model)

# ----------------------------
# Training loop
# ----------------------------
epochs_per_frame = 10
total_frames = 10
data = [(X_tensor, Y_tensor)]  # single batch
model_snapshots = Flux.Chain[]

for frame in 0:total_frames
    for epoch in 1:epochs_per_frame
        Flux.train!(loss_fn, model, data, opt_state)
    end
    println("Frame $frame, Loss: ", loss_fn(model, X_tensor, Y_tensor))
    push!(model_snapshots, deepcopy(model))

    # CAB analysis
    #CAB_analysis.calculate_quadratic_CAB_partition_tree(model, 3, 1, frame)
    CAB_analysis.calculate_quadratic_CAB_neuron_table(model, frame)
end

# ----------------------------
# CAB_analysis visualization
# ----------------------------
#CAB_analysis.get_CAB_partition_tree(100)
#CAB_analysis.plot_quadratic_CAB(model_snapshots, 3, 1, 1)
#CAB_analysis.plot_empirical_CAB(model_snapshots, 3, 1, 1)

CAB_analysis.create_quadratic_animation(model_snapshots, total_frames)
#CAB_analysis.create_animation_2(n, total_frames)
CAB_analysis.create_empirical_animation(model_snapshots, total_frames)
#CAB_analysis.plot_partition_count(total_frames)
CAB_analysis.create_phi_animation(model_snapshots, total_frames)
CAB_analysis.plot_CAB_changes(model_snapshots, total_frames)
# ----------------------------
# Decision boundary plot
# ----------------------------
x1_range = range(minimum(X[:,1])-0.5, stop=maximum(X[:,1])+0.5, length=300)
x2_range = range(minimum(X[:,2])-0.5, stop=maximum(X[:,2])+0.5, length=300)
grid_points = [[x1, x2] for x1 in x1_range, x2 in x2_range]  # x1 outer, x2 inner
grid_matrix = hcat(vec.(grid_points)...)
probs = σ.(model(grid_matrix))
probs_matrix = reshape(probs, (length(x1_range), length(x2_range)))'

probs = σ.(model(grid_matrix))
probs_matrix = reshape(probs, (length(x2_range), length(x1_range)))

f = Figure(size=(500,500))
ax = Axis(f[1,1])
contourf!(ax, x1_range, x2_range, probs_matrix, colormap=:RdBu)
scatter!(ax, X[:,1], X[:,2], color=Y[:,1], colormap=:Spectral)
ax.title = "Moons Decision Boundary (After Training)"
ax.xlabel = "x₁"
ax.ylabel = "x₂"
save("plot_store/moons_decision_boundary.png", f)


