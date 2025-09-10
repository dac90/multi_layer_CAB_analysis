##############################
# CAB_analysis.jl
##############################

module CAB_analysis

# === Import Packages ===
using PyCall
using Serialization
using Statistics
using LinearAlgebra
using JuMP
using HiGHS
using GLMakie
using GraphMakie
using Graphs
using Colors
using ColorSchemes
using LaTeXStrings
using Printf
using ColorTypes
using ColorVectorSpace
using ImageCore
using FFMPEG
using Flux
using GeometryBasics
using Meshes
using Ipopt
import Contour

# Include other files
include("CAB_visualiser_empirical.jl")
using .CAB_visualiser_empirical
export plot_empirical_CAB, create_empirical_animation, plot_empirical_CAB_3D

# === Export Functions === #
export calculate_CAB_partition_tree, calculate_CAB_boundary_tree, calculate_CAB_neuron_table, plot_CAB, create_animation, create_animation_2
export get_CAB_partition_tree, get_CAB_neuron_table
export get_partition_count, plot_partition_count, plot_partition_count_quantiles
export calculate_quadratic_CAB_partition_tree, calculate_quadratic_CAB_neuron_table, plot_quadratic_CAB, create_quadratic_animation

# === Pre-Defined Variables === #
cmap_boundary = reverse(cgrad(ColorSchemes.bam, 256))
cmap_null     = reverse(cgrad(ColorSchemes.vik, 256))
cmap_nonbound = reverse(cgrad(ColorSchemes.broc, 256))

# === Feasability Model Structure === #
mutable struct FeasibilityModel
    optimisation_model::Model
    x::Vector{VariableRef}
    epsilon::Float64
end

# === Partition Data Structure === #
struct LinearPartitionEntry
    phi::Vector{Float64}
    pattern::BitVector
    W_hat::Matrix{Float64}
    b_hat::Vector{Float64}
    W_tilde::Matrix{Float64}
    b_tilde::Vector{Float64}
    super_partition::Union{UInt128, Nothing}
    sub_partitions::Vector{UInt128}
    tag::String
end

struct QuadraticPartitionEntry
    pattern::BitVector
    Q::Matrix{Float64}
    Q_tilde::Vector{Matrix{Float64}}
    super_partition::Union{UInt128, Nothing}
    sub_partitions::Vector{UInt128}
    tag::String
end

# === Plotting Grid === #
struct ConstGrid
    x::LinRange{Float64}
    y::LinRange{Float64}
    Xg::Matrix{Float64}
    Yg::Matrix{Float64}
    points::Matrix{Float64}
    points_aug::Matrix{Float64}
    points_T::Matrix{Float64}
    points_aug_T::Matrix{Float64}
    n_points::Int
end

const CAB_GRID = let
    res = 500
    x = LinRange(-5, 5, res)
    y = LinRange(-5, 5, res)
    Xg = repeat(reshape(x, :, 1), 1, length(y))
    Yg = repeat(reshape(y, 1, :), length(x), 1)
    points = hcat(vec(Xg), vec(Yg))
    points_aug = hcat(points, ones(size(points, 1)))
    points_T = points'
    points_aug_T = points_aug'
    n_points = res * res
    ConstGrid(x, y, Xg, Yg, points, points_aug, points_T, points_aug_T, n_points)
end

function get_n(model::Flux.Chain, input_dim::Int)
    n = [input_dim]
    current_dim = input_dim
    
    for l in model.layers
        if l isa Dense
            current_dim = size(l.weight, 1)  # number of output neurons
            push!(n, current_dim)
        elseif l isa LayerNorm || l isa BatchNorm
            push!(n, current_dim)
        end
        # skip relu / identity layers, since they are absorbed
    end
    return n
end

function get_z(model::Flux.Chain, a::Matrix{Float64})
    pre_activations = Vector{Dict}()  # start empty
    for layer in model
        if layer isa Dense
            z = layer.weight * a .+ layer.bias
            a = layer.σ.(z)
        elseif layer isa BatchNorm
            norm_a = (a .- layer.:μ) ./ sqrt.(layer.:σ² .+ layer.ϵ)
            z = layer.:γ .* norm_a .+ layer.:β
            a = layer.λ.(z)
        elseif layer isa LayerNorm
            layer_mean_a = mean(a; dims=1)
            layer_var_a = mean((a .- layer_mean_a).^2; dims=1)
            norm_a = (a .- layer_mean_a) ./ sqrt.(layer_var_a .+ layer.ϵ)
            z = (layer.:diag.scale .* norm_a) .+ layer.:diag.bias
            a = layer.λ.(z)
        end
        push!(pre_activations, Dict(:z => z, :a => a))
    end
    return pre_activations
end

"""
    all_activation_patterns(sizes::Vector{Int}) -> Iterator{Vector{BitVector}}

Generates all possible ReLU activation patterns for each layer size.
Returns an iterator of activation patterns as Vector{BitVector}.
"""
function all_activation_patterns(n_l::Int)
    m = 2^n_l
    patterns = Vector{BitVector}(undef, m)
    for i in 0:m-1
        bv = falses(n_l)
        for j in 1:n_l
            bv[j] = (i >> (n_l - j)) & 1 == 1
        end
        patterns[i+1] = bv
    end
    return patterns
end

"""
    calculate_CAB_partition_tree(model::Flux.Chain, neuron_layer::Int, neuron_index::Int, frame::Union{Int, Nothing} = nothing, to_save::Bool = true)

Calculates the CAB of a Neuron in all lower layers, latent and otherwise. Ignores void partitions effectively
"""

function calculate_CAB_partition_tree(model::Flux.Chain, neuron_layer::Int, neuron_index::Int, frame::Union{Int, Nothing} = nothing, to_save::Bool = true)

    function init_fm(x_size::Int; solver=HiGHS.Optimizer, epsilon=1e-3)
        optimisation_model = Model(solver)
        set_silent(optimisation_model)
        @variable(optimisation_model, x[1:x_size])
        return FeasibilityModel(optimisation_model, x, epsilon)
    end

    function LP_feasibility(fm::FeasibilityModel, A::Matrix{Float64}, b::Vector{Float64}, pattern::BitVector)
        # Input variables
        orthant = 2 .* Int.(pattern) .- 1
        A_flipped = Diagonal(orthant) * A
        b_flipped = b .* orthant

        # Add constraints: W_scaled * x + b_scaled >= epsilon (small positive)
        epsilon = 1e-3
        con_refs = @constraint(fm.optimisation_model, A_flipped * fm.x .+ b_flipped .>= epsilon)

        optimize!(fm.optimisation_model)
        status = termination_status(fm.optimisation_model)

        # Delete constraints for next iteration
        foreach(c -> delete(fm.optimisation_model, c), con_refs)
        return status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
    end

    n = get_n(model, 2)
    CAB_partition_tree = Vector{Dict{UInt128, LinearPartitionEntry}}(undef, neuron_layer)
    partition_layer = Dict{UInt128, LinearPartitionEntry}()

    W_init = model[neuron_layer].weight[neuron_index:neuron_index, :]
    b_init = model[neuron_layer].bias[neuron_index:neuron_index]
    phi_init = -pinv(W_init) * b_init
    partition_layer[0] = LinearPartitionEntry(phi_init, Vector{BitVector}(), W_init, b_init, Matrix{Float64}(undef, 0, size(W_init)[2]), Vector{Float64}(), nothing, Vector{UInt128}(), "Boundary")
    CAB_partition_tree[1] = partition_layer

    for l in neuron_layer-2:-1:0
        fm1 = init_fm(n[l+1])
        fm2 = init_fm(n[l+1]-1)
        partition_layer = Dict{UInt128, LinearPartitionEntry}()
        partition_key = 0
        for (super_partition_key, super_partition) in CAB_partition_tree[neuron_layer-l-1]
            for layer_pattern in all_activation_patterns(n[l+2])
                pattern =  vcat(layer_pattern, super_partition.pattern)

                P = Diagonal(layer_pattern)
                W = model[l+1].weight
                b = model[l+1].bias

                W_hat = super_partition.W_hat * P * W
                b_hat = super_partition.b_hat + (super_partition.W_hat * P * b)
                W_tilde = vcat(W, super_partition.W_tilde * P * W)
                b_tilde = vcat(b, super_partition.b_tilde + (super_partition.W_tilde * P * b))

                phi = -pinv(W_hat) * b_hat
                nonvoid_bool = LP_feasibility(fm1, W_tilde, b_tilde, pattern)

                tag = "Void"
                if nonvoid_bool
                    if all(phi .== 0.0)
                        tag = "Null"
                    else    # If non-void and non-null, perform a projection from the space of the CAB plane and repeat LP test to test for boundary region
                        Q, _ = qr([phi I])
                        phi_ortho = Q[:, 2:end]
                        W_tilde_proj = W_tilde * phi_ortho
                        b_tilde_proj = (W_tilde * phi) + b_tilde
                        boundary_bool = LP_feasibility(fm2, W_tilde_proj, b_tilde_proj, pattern)
                        if boundary_bool
                            tag = "Boundary"
                        else
                            tag = "Non-Boundary"
                        end
                    end
                    partition_layer[partition_key] = LinearPartitionEntry(phi, pattern, W_hat, b_hat, W_tilde, b_tilde, super_partition_key, Vector{UInt128}(), tag)
                    push!(super_partition.sub_partitions, partition_key)
                    partition_key += 1
                end
            end
        end
        CAB_partition_tree[neuron_layer - l] = partition_layer
        empty!(fm1.optimisation_model)
        empty!(fm2.optimisation_model)
    end
    if to_save
        if isnothing(frame)
            save_path = "data/CAB_partition_tree.jlser"
        else
            save_path = @sprintf("data/CAB_partition_tree_%04d.jlser", frame)
        end
        println("Saving CAB tree (including non-boundary) to $save_path using Serialization")
        open(save_path, "w") do io
            serialize(io, CAB_partition_tree)
        end
    end
    return CAB_partition_tree
end

"""
    calculate_CAB_boundary_tree(n::Vector{Int}, neuron_index::Int, frame::Union{Int, Nothing} = nothing)

Calculates the CAB of a Neuron in all lower layers, latent and otherwise. Ignores void and non-boundary partitions effectively
"""

function calculate_CAB_boundary_tree(model::Flux.Chain, neuron_layer::Int, neuron_index::Int, frame::Union{Int, Nothing} = nothing, to_save::Bool=true)

    function init_fm(x_size::Int; solver=HiGHS.Optimizer, epsilon = 1e-3)
        optimisation_model = Model(solver)
        set_silent(optimisation_model)
        @variable(optimisation_model, x[1:x_size])
        return FeasibilityModel(optimisation_model, x, epsilon)
    end

    function LP_feasibility(fm::FeasibilityModel, A::Matrix{Float64}, b::Vector{Float64}, pattern::BitVector)
        # Input variables
        orthant = 2 .* Int.(pattern) .- 1
        A_flipped = Diagonal(orthant) * A
        b_flipped = b .* orthant

        # Add constraints: W_scaled * x + b_scaled >= epsilon (small positive)
        epsilon = 1e-3
        con_refs = @constraint(fm.optimisation_model, A_flipped * fm.x .+ b_flipped .>= epsilon)

        optimize!(fm.optimisation_model)
        status = termination_status(fm.optimisation_model)

        # Delete constraints for next iteration
        foreach(c -> delete(fm.optimisation_model, c), con_refs)
        return status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
    end

    n = get_n(model, 2)
    CAB_partition_tree = Vector{Dict{UInt128, LinearPartitionEntry}}(undef, neuron_layer)
    partition_layer = Dict{UInt128, LinearPartitionEntry}()
    
    W_init = model[neuron_layer].weight[neuron_index:neuron_index, :]
    b_init = model[neuron_layer].bias[neuron_index:neuron_index]
    phi_init = -pinv(W_init) * b_init
    partition_layer[0] = LinearPartitionEntry(phi_init, Vector{BitVector}(), W_init, b_init, Matrix{Float64}(undef, 0, size(W_init)[2]), Vector{Float64}(), nothing, Vector{UInt128}(0), "Boundary")
    CAB_partition_tree[1] = partition_layer

    for l in neuron_layer-2:-1:0
        fm = init_fm(n[l+1]-1)
        partition_layer = Dict{UInt128, LinearPartitionEntry}()
        partition_key = 0
        for (_, super_partition) in CAB_partition_tree[neuron_layer-l-1]
            for layer_pattern in all_activation_patterns(n[l+2])
                pattern =  vcat(layer_pattern, super_partition.pattern)

                P = Diagonal(layer_pattern)
                W = model[l+1].weight
                b = model[l+1].bias

                W_hat = super_partition.W_hat * P * W
                b_hat = super_partition.b_hat + (super_partition.W_hat * P * b)
                W_tilde = vcat(W, super_partition.W_tilde * P * W)
                b_tilde = vcat(b, super_partition.b_tilde + (super_partition.W_tilde * P * b))

                phi = -pinv(W_hat) * b_hat
                if !all(phi .== 0.0) # If non-void and non-null, perform a projection from the space of the CAB plane and repeat LP test to test for boundary region
                    Q, _ = qr([phi I])
                    phi_ortho = Q[:, 2:end]
                    W_tilde_proj = W_tilde * phi_ortho
                    b_tilde_proj = (W_tilde * phi) + b_tilde
                    boundary_bool = LP_feasibility(fm, W_tilde_proj, b_tilde_proj, pattern)
                    if boundary_bool
                        partition_layer[partition_key] = LinearPartitionEntry(phi, pattern, W_hat, b_hat, W_tilde, b_tilde, super_partition_key, Vector{UInt128}(), tag)
                        push!(super_partition.sub_partitions, partition_key)
                        partition_key += 1
                    end
                end
            end
        end
        CAB_partition_tree[neuron_layer - l] = partition_layer
        empty!(fm.optimisation_model)
    end

    if to_save
        if isnothing(frame)
            save_path = "data/CAB_partition_tree.jlser"
        else
            save_path = @sprintf("data/CAB_partition_tree_%04d.jlser", frame)
        end
        println("Saving CAB tree (boundary only) to $save_path using Serialization")
        open(save_path, "w") do io
            serialize(io, CAB_partition_tree)
        end
    end
    return CAB_partition_tree
end

"""
    get_CAB_partition_tree(path::String)

Deserializes the `.jlser` file at `path`, prints each partition's CAB position vector and tag
"""
function get_CAB_partition_tree(frame::Int)
    load_path = @sprintf("data/CAB_partition_tree_%04d.jlser", frame)
    CAB_partition_tree = deserialize(load_path)
    for partition_layer in CAB_partition_tree
        for k in sort(collect(keys(partition_layer)))
            v = partition_layer[k]
            println("$k => phi: ", v.phi, ", tag: ", v.tag)
        end
    end
    return CAB_partition_tree
end

function graph_CAB_partition_tree!(ax::Axis, frame::Int; show_labels::Bool = true)
    # --- Load data ---
    load_path = @sprintf("data/CAB_partition_tree_%04d.jlser", frame)
    CAB_partition_tree = deserialize(load_path)

    # --- Flatten all layers into one dict ---
    all_parts = Dict{UInt128, LinearPartitionEntry}()
    for d in CAB_partition_tree
        merge!(all_parts, d)
    end

    # --- Map UInt128 keys to graph vertex indices ---
    keys_vec = collect(keys(all_parts))
    key_to_idx = Dict(k => i for (i,k) in enumerate(keys_vec))

    # --- Build edge list ---
    edges = Tuple{Int,Int}[]
    for (key, entry) in all_parts
        for child in entry.sub_partitions
            push!(edges, (key_to_idx[key], key_to_idx[child]))
        end
    end

    g = Graph(length(keys_vec))
    for (src, dst) in edges
        add_edge!(g, src, dst)
    end

    # --- Assign coordinates, centered above children ---
    coords = Dict{UInt128, Point2f}()

    function assign_positions(key::UInt128, depth::Int, ypos::Float64)
        entry = all_parts[key]
        children = entry.sub_partitions
        if isempty(children)
            coords[key] = Point2f(-depth, ypos)
            return ypos - 1   # decrement to move downwards
        else
            start_y = ypos
            for child in children
                ypos = assign_positions(child, depth+1, ypos)
            end
            ys = [coords[c][2] for c in children]
            coords[key] = Point2f(-depth, mean(ys))
            return ypos
        end
    end

    # Find root (no super_partition)
    root_key = only([k for (k,v) in all_parts if v.super_partition === nothing])
    num_leaves = count(v -> isempty(v.sub_partitions), values(all_parts))
    max_y = num_leaves + 1.0  # add some padding
    assign_positions(root_key, 0, max_y)

    layout = [coords[k] for k in keys_vec]

    # --- Tag-based colors ---
    function tagcolor(tag)
        tag == "Boundary"      && return :red
        tag == "Non-Boundary"  && return :blue
        tag == "Null"          && return :purple
        return :gray
    end
    node_colors = [tagcolor(all_parts[k].tag) for k in keys_vec]

    # --- Short hex display ---
    shorthex(k::UInt128) = "0x" * uppercase(string(k, base=16, pad=2)[end-1:end])

    # --- Node labels: key + phi ---
    if show_labels
        node_labels = [string("ϕ=", join(round.(all_parts[k].phi, digits=3), ",")) for k in keys_vec]
    else
        node_labels = fill("", length(keys_vec))  # same length, but empty labels
    end

    # --- Plot ---
    graphplot!(
        ax, g;
        layout = layout,
        node_size = 20,
        node_color = node_colors,
        nlabels = node_labels,
        nlabels_color = :black,
        nlabels_align = (:center, :bottom),
        edge_color = :black,
        edge_width = 2
    )
    hidedecorations!(ax)
    hidespines!(ax)
end


"""
    calculate_CAB_neuron_table()

Calculates the partitions in the input layer only, but it does so for all neurons in the network
"""

function calculate_CAB_neuron_table(model::Flux.Chain, frame::Union{Int, Nothing} = nothing, to_save::Bool = true)
    n = get_n(model, 2)
    L = size(n)[1] - 1
    partition_neuron_table = Vector{Vector{Dict{UInt128, LinearPartitionEntry}}}(undef, L)
    for l in 1:L
        partition_neuron_layer = Vector{Dict{UInt128, LinearPartitionEntry}}(undef, n[l+1])
        for i in 1:n[l+1]
            partition_neuron_layer[i] = calculate_CAB_partition_tree(model, l, i, frame, false)[end]
        end
        partition_neuron_table[l] = partition_neuron_layer
    end

    if to_save
        if isnothing(frame)
            save_path = "data/partition_neuron_table.jlser"
        else
            save_path = @sprintf("data/partition_neuron_table_%04d.jlser", frame)
        end
        println("Saving CAB neuron table (including non-boundary) to $save_path using Serialization")
        open(save_path, "w") do io
            serialize(io, partition_neuron_table)
        end
    end
    return partition_neuron_table
end

"""
    get_CAB_neuron_table(path::String)

Deserializes the `.jlser` file at `path`, prints each neuron's CAB position vector and tag for each partition
"""

function get_CAB_neuron_table(frame)
    # --- Load File ---
    if isnothing(frame)
        load_path = "data/partition_neuron_table.jlser"
    else
        load_path = @sprintf("data/partition_neuron_table_%04d.jlser", frame)
    end
    partition_neuron_table = deserialize(load_path)

    # --- Output Data ---
    for partition_neuron_layer in partition_neuron_table
        for partition_layer in partition_neuron_layer
            for k in sort(collect(keys(partition_layer)))
                v = partition_layer[k]
                println("$k => phi: ", v.phi, ", tag: ", v.tag)
            end
        end
    end

    return partition_neuron_table
end

function plot_CAB_frame!(ax::Axis, neuron_layer::Int, neuron_index::Int, partition_neuron_table:: Vector{Vector{Dict{UInt128, LinearPartitionEntry}}}, frame::Int)
    # --- Data Setup ---
    x, y = CAB_GRID.x, CAB_GRID.y
    points = CAB_GRID.points
    points_T = CAB_GRID.points_T
    grid_size = size(CAB_GRID.Xg) 

    mask = falses(CAB_GRID.n_points)
    z = Vector{Float64}(undef, CAB_GRID.n_points)

    # --- Clear Previous Plot ---
    empty!(ax.scene.plots)
    ax.title = isnothing(frame) ? latexstring("z^{[$neuron_layer]}_$neuron_index \\text{ with CAB}") : latexstring("z^{[$neuron_layer]}_{$neuron_index} \\text{ with CAB at frame $frame}")
    ax.xlabel = latexstring("a^{[0]}_1")
    ax.ylabel = latexstring("a^{[0]}_2")
    println("Creating Figure for Neuron Layer: $neuron_layer, Neuron Index: $neuron_index, Frame: $frame")

    # === Plot Outputs === #

    color_limits = (-10, 10)
    colors = [get(ColorSchemes.turbid, i/(length(partition_neuron_table)-1)) for i in 0:(length(partition_neuron_table)-1)]

    Zg_boundary     = fill(NaN, grid_size)
    Zg_null         = fill(NaN, grid_size)
    Zg_non_boundary = fill(NaN, grid_size)

    for (_, partition) in partition_neuron_table[neuron_layer][neuron_index] # For each partition in the chosen neuron
        # --- Generate Mask ---
        fill!(mask, false)
        if isempty(partition.W_tilde) || isempty(partition.b_tilde)
            mask .= true
        else
            orthant = 2 .* Int.(pattern) .- 1
            W_tilde_flipped = Diagonal(orthant) * partition.W_tilde
            b_tilde_flipped = partition.b_tilde .* orthant
            mask .= vec(all(W_tilde_flipped * points_T .+ b_tilde_flipped .> 0, dims=1))
        end

        # --- Generate Outputs ---
        mul!(z, points, vec(partition.W_hat))
        z .+= partition.b_hat

        # --- Add to Appropriate Type ---
        if partition.tag == "Boundary"
            Zg_boundary[mask] .= z[mask]
        elseif partition.tag == "Null"
            Zg_null[mask] .= z[mask]
        else
            Zg_non_boundary[mask] .= z[mask]
        end
    end

    # --- Plot Heatmaps ---
    heatmap!(ax, x, y, Zg_boundary; colormap = reverse(cgrad(ColorSchemes.bam)), colorrange = color_limits, alpha=1)
    heatmap!(ax, x, y, Zg_null;     colormap = reverse(cgrad(ColorSchemes.vik)), colorrange = color_limits, alpha=1)
    heatmap!(ax, x, y, Zg_non_boundary; colormap = reverse(cgrad(ColorSchemes.broc)), colorrange = color_limits, alpha=1)

    # === Plot CAB === #

    spacing = 0.05 # = step(LinRange(-5, 5, 200))
    max_length = 7.5  # = √2 * 5
    
    for (i, partition_neuron_layer) in enumerate(partition_neuron_table[1:neuron_layer]) # Each layer
        for (j, partition_neuron) in enumerate(partition_neuron_layer) # Each neuron
            if i != neuron_layer || j == neuron_index # Only plot one top-layer neuron
                for (_, partition) in partition_neuron # Each partition
                    if partition.tag == "Boundary"
                        # --- Generate CAB Points ---
                        perp_phi = vec([-partition.phi[2], partition.phi[1]])
                        CAB_draw_step = perp_phi / norm(perp_phi)
                        CAB_step_range = vec(collect(-max_length:spacing:max_length))
                        CAB_points = (CAB_step_range * CAB_draw_step') .+ partition.phi'
                        
                        # --- Mask for Partition
                        CAB_points_mask = all((-5 .<= CAB_points .<= 5), dims=2)[:]
                        if !isempty(partition.pattern)
                            orthant = 2 .* Int.(pattern) .- 1
                            W_tilde_flipped = Diagonal(orthant) * partition.W_tilde
                            b_tilde_flipped = partition.b_tilde .* orthant
                            CAB_points_mask .&= vec(all(W_tilde_flipped * CAB_points' .+ b_tilde_flipped .> -1e-2, dims = 1))
                        end

                        # --- Plot ---
                        if any(CAB_points_mask)
                            first_idx = findfirst(CAB_points_mask)
                            last_idx = findlast(CAB_points_mask)
                            xs = [CAB_points[first_idx, 1], CAB_points[last_idx, 1]]
                            ys = [CAB_points[first_idx, 2], CAB_points[last_idx, 2]]
                            lines!(ax, xs, ys; linewidth = 2, color = colors[i])
                        end
                    end
                end
            end
        end
    end
end

"""
    plot_CAB(weights::Dict, n::Vector{Int})

Plots all activation boundaries for a 2D input network.
"""

function plot_CAB(n::Vector{Int}, neuron_layer::Int, neuron_index::Int, frame::Union{Int, Nothing} = nothing, to_save::Bool = true)
    # --- Load File ---
    if isnothing(frame)
        load_path = "data/partition_neuron_table.jlser"
    else
        load_path = @sprintf("data/partition_neuron_table_%04d.jlser", frame)
    end
    partition_neuron_table = deserialize(load_path)
    
    # --- Figure Setup ---
    title_text = isnothing(frame) ? latexstring("z^{[$neuron_layer]}_$neuron_index \\text{ with CAB}") : latexstring("z^{[$neuron_layer]}_{$neuron_index} \\text{ with CAB at frame $frame}")
    fig = Figure(size = (900, 600))
    ax = Axis(fig[1, 1],
        title = title_text,
        xlabel = L"x_1",
        ylabel = L"x_2",
        aspect = DataAspect(),
        limits = ((-5, 5), (-5, 5))
    )

    # --- Call shared plotting logic ---
    plot_CAB_frame!(ax, neuron_layer, neuron_index, partition_neuron_table, frame)

    # --- Legend for Layers (recomputed here for fig) ---
    colors = [get(ColorSchemes.turbid, i/(length(n)-2)) for i in 0:(length(n)-2)]

    for (layer_idx, col) in enumerate(colors)
        label_idx = length(colors) - layer_idx + 1
        lines!(ax, [NaN], [NaN], color=col, linewidth=2, label="Layer $label_idx")
    end
    axislegend(ax, position=:lt)

    # --- Colorbars ---
    subgrid = fig[1, 2] = GridLayout()
    Colorbar(subgrid[1, 1], colormap = reverse(cgrad(ColorSchemes.bam)), limits = (-10, 10), label = "Boundary", width = 20, height = Relative(0.9))
    Colorbar(subgrid[1, 2], colormap = reverse(cgrad(ColorSchemes.vik)), limits = (-10, 10), label = "Null", width = 20, height = Relative(0.9))
    Colorbar(subgrid[1, 3], colormap = reverse(cgrad(ColorSchemes.broc)), limits = (-10, 10), label = "Non-Boundary", width = 20, height = Relative(0.9))

    # --- Save Figure ---
    if to_save
        if isnothing(frame)
            save_path = "CAB_plot.png"
        else
            save_path = @sprintf("plot_store/CAB_plot_%04d.png", frame)
        end
        println("Saving CAB plot to $save_path")
        save(save_path, fig)
    end

    return fig
end

function create_animation(n::Vector{Int}, total_frame::Int; output_path::String = "CAB_animation.mp4", framerate::Int = 10)
    # === Figure ===
    fig = Figure(size = (1280, 720))
    rowgap!(fig.layout, 5)         # 5px vertical spacing
    colgap!(fig.layout, 5)         # 5px horizontal spacing

    # === Title ===
    Label(fig[0, 2], "Pre-activations and CAB's of all neurons", fontsize = 28, tellwidth = false, tellheight = true)
    rowsize!(fig.layout, 0, Auto(false))

    # === Legend for Layers ===
    colors = [get(ColorSchemes.turbid, i/(length(n)-2)) for i in 0:(length(n)-2)]
    dummy_axis = Axis(fig.scene)  # Create axis not added to layout
    legend_lines = [lines!(dummy_axis, [NaN], [NaN], color=col, linewidth=2) for col in colors]
    Legend(fig[1, 1], legend_lines, ["Layer $i" for i in 1:length(colors)]; title = "CAB Legend")
    colsize!(fig.layout, 1, Auto(false))

    # === Colorbars === 
    subgrid = fig[1, 3] = GridLayout()
    colgap!(subgrid, 2)
    Colorbar(subgrid[1, 1], colormap = reverse(cgrad(ColorSchemes.bam)), limits = (-10, 10), label = "Boundary", width = 10, height = Relative(0.9), flip_vertical_label = true)
    Colorbar(subgrid[1, 2], colormap = reverse(cgrad(ColorSchemes.vik)), limits = (-10, 10), label = "Null", width = 10, height = Relative(0.9), flip_vertical_label = true)
    Colorbar(subgrid[1, 3], colormap = reverse(cgrad(ColorSchemes.broc)), limits = (-10, 10), label = "Non-Boundary", width = 10, height = Relative(0.9), flip_vertical_label = true)

    colsize!(subgrid, 1, Auto(false))
    colsize!(subgrid, 2, Auto(false))
    colsize!(subgrid, 3, Auto(false))
    colsize!(fig.layout, 3, Auto(false))

    # === Plots ===
    axes_grid = Vector{Vector{Axis}}(undef, length(n)-1)
    for row in 1:length(n)-1
        neuron_layer = length(n) - row
        
        row_grid = fig[row, 2] = GridLayout()  
        colgap!(row_grid, 5)
        rowsize!(fig.layout, row, Auto(false))

        axes_grid[row] = Vector{Axis}(undef, n[neuron_layer + 1])
        for neuron_index in 1:n[neuron_layer+1]
            axes_grid[row][neuron_index] = Axis(row_grid[1, neuron_index], aspect = DataAspect())
            colsize!(row_grid, neuron_index, Auto(false))
        end
    end
    rowsize!(fig.layout, 1, Relative(0.4))
    colsize!(fig.layout, 2, Relative(0.8))

    # === Animation Recording ===
    record(fig, output_path, 0:total_frame; framerate = framerate) do frame
        partition_neuron_table = deserialize(@sprintf("data/partition_neuron_table_%04d.jlser", frame))
        for row in 1:length(n)-1
            neuron_layer = length(n) - row
            for neuron_index in 1:n[neuron_layer+1]
                ax = axes_grid[row][neuron_index]
                empty!(ax)
                plot_CAB_frame!(ax, neuron_layer, neuron_index, partition_neuron_table, frame)
            end
        end
    end

    println("Animation saved to $output_path") # Note that things are saved live in the previous block
end

function create_animation_2(n::Vector{Int}, total_frame::Int; output_path::String = "CAB_animation_2.mp4", framerate::Int = 10)
    # === Figure ===
    fig = Figure(size = (1280, 720))
    rowgap!(fig.layout, 5)         # 5px vertical spacing
    colgap!(fig.layout, 5)         # 5px horizontal spacing

    # === Title ===
    Label(fig[0, 2], "Pre-activations and CAB's of all neurons", fontsize = 28, tellwidth = false, tellheight = true)
    rowsize!(fig.layout, 0, Auto(false))

    # === Legend for Layers ===
    colors = [get(ColorSchemes.turbid, i/(length(n)-2)) for i in 0:(length(n)-2)]
    dummy_axis = Axis(fig.scene)  # Create axis not added to layout
    legend_lines = [lines!(dummy_axis, [NaN], [NaN], color=col, linewidth=2) for col in colors]
    Legend(fig[1, 1], legend_lines, ["Layer $i" for i in 1:length(colors)]; title = "CAB Legend")
    colsize!(fig.layout, 1, Auto(false))

        # === Plots ===
    ax_CAB = Axis(fig[1, 2], aspect = DataAspect())
    rowsize!(fig.layout, 1, Auto(false))
    colsize!(fig.layout, 2, Relative(0.8))

    # === Colorbars === 
    subgrid = fig[1, 3] = GridLayout()
    colgap!(subgrid, 2)
    Colorbar(subgrid[1, 1], colormap = reverse(cgrad(ColorSchemes.bam)), limits = (-10, 10), label = "Boundary", width = 10, height = Relative(0.9), flip_vertical_label = true)
    Colorbar(subgrid[1, 2], colormap = reverse(cgrad(ColorSchemes.vik)), limits = (-10, 10), label = "Null", width = 10, height = Relative(0.9), flip_vertical_label = true)
    Colorbar(subgrid[1, 3], colormap = reverse(cgrad(ColorSchemes.broc)), limits = (-10, 10), label = "Non-Boundary", width = 10, height = Relative(0.9), flip_vertical_label = true)

    colsize!(subgrid, 1, Auto(false))
    colsize!(subgrid, 2, Auto(false))
    colsize!(subgrid, 3, Auto(false))
    colsize!(fig.layout, 3, Auto(false))

    # === Partition Tree Graph ===
    ax_tree = Axis(fig[2, 2])
    rowsize!(fig.layout, 2, Relative(0.6))

    # === Animation Recording ===
    record(fig, output_path, 0:total_frame; framerate = framerate) do frame
        partition_neuron_table = deserialize(@sprintf("data/partition_neuron_table_%04d.jlser", frame))
        empty!(ax_CAB)
        plot_CAB_frame!(ax_CAB, length(n)-1, 1, partition_neuron_table, frame)
        empty!(ax_tree)
        graph_CAB_partition_tree!(ax_tree, frame; show_labels = false)
    end

    println("Animation saved to $output_path") # Note that things are saved live in the previous block
end

function get_partition_count(model::Flux.Chain, n::Vector{Int}, neuron_index::Int)
    CAB_partition_tree = calculate_CAB_partition_tree(model, n, neuron_index, nothing, false)
    vals = values(CAB_partition_tree[end])
    n_boundary     = count(s -> s.tag == "Boundary", vals)
    n_null         = count(s -> s.tag == "Null", vals)
    n_non_boundary  = count(s -> s.tag == "Non-Boundary", vals)
    return(n_boundary, n_null, n_non_boundary)
end

"""
    plot_partition_count(total_frames::Int, to_save = true) -> fig with 2 plots

    One plot of number of partitions of the input space at each frame, another of the number of boundary partitions at each frame
"""
function plot_partition_count(boundary_count::Matrix{Int}, null_count::Matrix{Int}, non_boundary_count::Matrix{Int}, loss::Matrix{Float64}; to_save::Bool = true)
    # ------------------------
    # Compute counts & proportions
    # ------------------------
    partition_count = boundary_count + null_count + non_boundary_count
    boundary_proportion = boundary_count ./ partition_count

    # Means and stds across cycles (dim=1)
    mean_boundary     = vec(mean(boundary_count, dims=1))
    std_boundary      = vec(std(boundary_count, dims=1))

    mean_null         = vec(mean(null_count, dims=1))
    std_null          = vec(std(null_count, dims=1))

    mean_nonboundary  = vec(mean(non_boundary_count, dims=1))
    std_nonboundary   = vec(std(non_boundary_count, dims=1))

    mean_partition    = vec(mean(partition_count, dims=1))
    std_partition     = vec(std(partition_count, dims=1))

    mean_boundary_prop = vec(mean(boundary_proportion, dims=1))
    std_boundary_prop  = vec(std(boundary_proportion, dims=1))

    mean_loss         = vec(mean(loss, dims=1))
    std_loss          = vec(std(loss, dims=1))

    nframes = length(mean_boundary)
    frames = 1:nframes

    # ------------------------
    # Colors
    # ------------------------
    colors = (:red, :purple, :blue, :grey, :green, :brown)

    # ------------------------
    # Figure setup
    # ------------------------
    fig = Figure(resolution=(1200,600))
    ax = [Axis(fig[i,j]) for i in 1:2, j in 1:3]

    # Helper function for plotting mean ± std
    function plot_with_std(ax::Axis, ymean, ystd, color, title)
        lines!(ax, frames, ymean, color=color, linewidth=2)
        band!(ax, frames, ymean-ystd, ymean+ystd, color=color, alpha=0.2)
        ax.title = title       # <-- fix here
        ax.xlabel = "Frame"
        ax.ylabel = "Count / Proportion / Loss"
    end

    # Plot all 6 series
    plot_with_std(ax[1,1], mean_boundary, std_boundary, :red, "Boundary Count")
    plot_with_std(ax[1,2], mean_null, std_null, :purple, "Null Count")
    plot_with_std(ax[1,3], mean_nonboundary, std_nonboundary, :blue, "Non-Boundary Count")
    plot_with_std(ax[2,1], mean_partition, std_partition, :grey, "Partition Count")
    plot_with_std(ax[2,2], mean_boundary_prop, std_boundary_prop, :green, "Boundary Proportion")
    plot_with_std(ax[2,3], mean_loss, std_loss, :brown, "Loss")

    # Optional save
    if to_save
        save("plot_store/partition_count.png", fig)
        println("Saved figure to plot_store/partition_count.png")
    end

    display(fig)
    return fig
end

function plot_partition_count_quantiles(boundary_count::Matrix{Int}, 
                              null_count::Matrix{Int}, 
                              non_boundary_count::Matrix{Int}, 
                              loss::Matrix{Float64}; 
                              to_save::Bool=true)

    partition_count = boundary_count + null_count + non_boundary_count
    boundary_proportion = boundary_count ./ partition_count

    data_list = [boundary_count, null_count, non_boundary_count, partition_count, boundary_proportion, loss]
    titles = ["Boundary Count", "Null Count", "Non-Boundary Count", 
              "Partition Count", "Boundary Proportion", "Loss"]
    colors = (:red, :purple, :blue, :grey, :green, :brown)

    nframes = size(boundary_count, 2)
    frames = 1:nframes

    fig = Figure(resolution=(1200,600))
    ax = [Axis(fig[i,j]) for i in 1:2, j in 1:3]

    function plot_quantiles(ax::Axis, data::Matrix, color::Symbol, title::String)
        # data: cycles × frames
        med = mapslices(x -> quantile(x, 0.5), data; dims=1) |> vec
        q25 = mapslices(x -> quantile(x, 0.25), data; dims=1) |> vec
        q75 = mapslices(x -> quantile(x, 0.75), data; dims=1) |> vec
        q5  = mapslices(x -> quantile(x, 0.05), data; dims=1) |> vec
        q10 = mapslices(x -> quantile(x, 0.10), data; dims=1) |> vec
        q90 = mapslices(x -> quantile(x, 0.90), data; dims=1) |> vec
        q95 = mapslices(x -> quantile(x, 0.95), data; dims=1) |> vec

        # ribbons: lightest to darkest
        band!(ax, frames, q5, q95, color=color, alpha=0.1)
        band!(ax, frames, q10, q90, color=color, alpha=0.15)
        band!(ax, frames, q25, q75, color=color, alpha=0.2)

        # median line
        lines!(ax, frames, med, color=color, linewidth=3, label="median")
        # optionally: thin lines for quartiles
        lines!(ax, frames, q25, color=color, linewidth=1, linestyle=:dash)
        lines!(ax, frames, q75, color=color, linewidth=1, linestyle=:dash)

        ax.title = title
        ax.xlabel = "Frame"
        ax.ylabel = "Value"
    end

    for i in 1:6
        plot_quantiles(ax[i], data_list[i], colors[i], titles[i])
    end

    if to_save
        save("plot_store/partition_count_quantiles.png", fig)
        println("Saved figure to plot_store/partition_count_quantiles.png")
    end

    display(fig)
    return fig
end

function calculate_quadratic_CAB_partition_tree(model::Flux.Chain, l_2::Int, neuron_index::Int, frame::Union{Int, Nothing} = nothing, to_save::Bool = true)

    function create_N_matrix(dim)
        return Matrix{Float64}(I, dim, dim) - (ones(dim, dim)/dim)
    end

    function normalise_matrix(A::Matrix{Float64})
        return (opnorm(A,2) > 0) ? A ./ opnorm(A,2) : A
    end

    function CAB_init_matrix(n_l::Int, i::Int)
        Q_out = zeros(n_l+1, n_l+1)
        Q_out[end, i] = 0.5
        Q_out[i, end] = 0.5
        return Q_out
    end

    function CAB_step_relu(Q_in::Matrix{Float64}, P::Diagonal{Bool, BitVector})
        Q_cal_out = P * Q_in[1:end-1,1:end-1] * P
        L_cal_out = P * Q_in[1:end-1, end]
        C_cal_out = Q_in[end, end]
        return [Q_cal_out L_cal_out; L_cal_out' C_cal_out]
    end

    function CAB_step_dense(Q_in::Matrix{Float64}, l::Int)
        dense_W = model[l].weight
        dense_b = model[l].bias

        Q_cal_in = Q_in[1:end-1,1:end-1]
        L_cal_in = Q_in[1:end-1, end]
        C_cal_in = Q_in[end, end]

        Q_cal_out = dense_W' * Q_cal_in * dense_W
        L_cal_out = (dense_W' * Q_cal_in * dense_b) + (dense_W' * L_cal_in)
        C_cal_out = (dense_b' * Q_cal_in * dense_b) + (2 * dense_b' * L_cal_in) + C_cal_in
        return normalise_matrix([Q_cal_out L_cal_out; L_cal_out' C_cal_out])
    end

    function CAB_step_batchnorm(Q_in::Matrix{Float64}, l::Int)
        batchnorm_gamma = Diagonal(model[l].:γ)
        batchnorm_beta = model[l].:β
        batchnorm_mu = model[l].:μ
        batchnorm_sigma = sqrt.(model[l].:σ² .+ model[l].:ϵ^2)
        batchnorm_inv_sigma = Diagonal(ones(length(batchnorm_sigma)) ./ batchnorm_sigma)

        Q_cal_in = Q_in[1:end-1,1:end-1]
        L_cal_in = Q_in[1:end-1, end]
        C_cal_in = Q_in[end, end]

        Q_cal_descaled = batchnorm_gamma * Q_cal_in * batchnorm_gamma
        L_cal_descaled = (batchnorm_gamma * Q_cal_in * batchnorm_beta) + (batchnorm_gamma * L_cal_in)
        C_cal_descaled = (batchnorm_beta' * Q_cal_in * batchnorm_beta) + (2 * batchnorm_beta' * L_cal_in) + C_cal_in
        
        Q_cal_out = batchnorm_inv_sigma * Q_cal_descaled * batchnorm_inv_sigma
        L_cal_out = (- batchnorm_inv_sigma * Q_cal_descaled * batchnorm_inv_sigma * batchnorm_mu) + (batchnorm_inv_sigma * L_cal_descaled)
        C_cal_out = (batchnorm_mu' * batchnorm_inv_sigma * Q_cal_descaled * batchnorm_inv_sigma * batchnorm_mu) - (2 * batchnorm_mu' * batchnorm_inv_sigma * L_cal_descaled) + C_cal_descaled

        return normalise_matrix([Q_cal_out L_cal_out; L_cal_out' C_cal_out])
    end

    function CAB_step_layernorm(Q_in::Matrix{Float64}, l::Int)
        layernorm_inv_W = pinv(2 * Q_in[1:end-1, end]' * Diagonal(model[l].:diag.scale))
        layernorm_b = ((2 * Q_in[1:end-1, end]' * model[l].:diag.bias) + Q_in[end, end])

        if all(N * layernorm_inv_W .== 0.0)
            Q_out = [zeros(n[l], n[l]) zeros(n[l]) ; zeros(n[l])' layernorm_b]
            Q_m_out = [zeros(n[l], n[l]) zeros(n[l]) ; zeros(n[l])' -1]
        elseif layernorm_b == 0
            Q_out = [zeros(n[l], n[l]) (N * layernorm_inv_W); (N * layernorm_inv_W)' 0]
            Q_m_out = [zeros(n[l], n[l]) zeros(n[l]) ; zeros(n[l])' -1]
        else
            Q_cal_out = (n[l] * N * layernorm_inv_W * layernorm_inv_W' * N) - ((layernorm_inv_W' * layernorm_inv_W)^2 * layernorm_b^2 * N)
            Q_out = [Q_cal_out zeros(n[l]); zeros(n[l])' 0]
            Q_m_out = [zeros(n[l],n[l]) (N * layernorm_inv_W * layernorm_b) ; (N * layernorm_inv_W * layernorm_b)' 0]                           
        end
        
        return normalise_matrix(Q_out), normalise_matrix(Q_m_out)
    end
    
    function init_fm(x_size::Int; epsilon = 1e-3, solver=Ipopt.Optimizer)
        optimisation_model = Model(solver)
        set_silent(optimisation_model)
        @variable(optimisation_model, x[1:x_size])
        return FeasibilityModel(optimisation_model, x, epsilon)
    end

    function QP_feasibility(fm::FeasibilityModel, A::Matrix{Float64}, A_tilde::Vector{Matrix{Float64}}, pattern::BitVector)
        # Convert orthant pattern to {-1, 1}
        orthant = 2 .* Int.(pattern) .- 1
        QP_n = length(fm.x)
        # Helper: build a fresh JuMP model with optional equality constraint
        function build_model(add_eq::Bool)
            QP_m = Model(Ipopt.Optimizer)
            set_silent(QP_m)
            # Decision variables
            @variable(QP_m, x[1:QP_n])
            # Slack variable for feasibility
            @variable(QP_m, slack >= 0)
            # Augmented vector [x; 1]
            aug_x = [x; 1.0]
            # Add inequality constraints with slack
            for QP_i in 1:length(A_tilde)
                mat = Symmetric(A_tilde[QP_i])
                @NLconstraint(QP_m, orthant[QP_i] * sum(aug_x[QP_k] * mat[QP_k,QP_l] * aug_x[QP_l] for QP_k=1:QP_n+1, QP_l=1:QP_n+1) + slack >= 0)
            end
            # Optional equality constraint as two-sided inequality with same slack
            if add_eq
                mat_eq = Symmetric(A)
                @NLconstraint(QP_m, sum(aug_x[QP_k] * mat_eq[QP_k,QP_l] * aug_x[QP_l] for QP_k=1:QP_n+1, QP_l=1:QP_n+1) - slack <= 0)
                @NLconstraint(QP_m, sum(aug_x[QP_k] * mat_eq[QP_k,QP_l] * aug_x[QP_l] for QP_k=1:QP_n+1, QP_l=1:QP_n+1) + slack >= 0)
            end
            # Objective: minimize slack
            @NLobjective(QP_m, Min, slack)
            return QP_m, slack
        end

        # Step 1: inequalities only
        QP_model_1, slack1 = build_model(false)
        optimize!(QP_model_1)
        δ1_val = value(slack1)
        feasible_without_eq = δ1_val <= fm.epsilon

        # Step 2: inequalities + equality
        feasible_with_eq = false
        if feasible_without_eq
            QP_model_2, slack2 = build_model(true)
            optimize!(QP_model_2)
            δ2_val = value(slack2)
            feasible_with_eq = δ2_val <= fm.epsilon
        end

        if feasible_without_eq
            if feasible_with_eq
                tag = "Boundary"
            else
                tag = "Non-Boundary"
            end
        else
            tag = "Void"
        end

        return feasible_without_eq, feasible_with_eq, tag
    end

    n = get_n(model, 2)
    CAB_partition_tree = Vector{Dict{UInt128, QuadraticPartitionEntry}}(undef, l_2)

    partition_layer = Dict{UInt128, QuadraticPartitionEntry}()
    Q_tilde = Vector{Matrix{Float64}}()
    layer_pattern = BitVector()
    layer = model[l_2]

    if layer isa Dense
        L_cal = layer.weight[neuron_index, :] / 2
        C_cal = layer.bias[neuron_index]
        Q = [zeros(n[l_2], n[l_2]) L_cal; L_cal' C_cal]

    elseif layer isa BatchNorm
        L_cal = zeros(n[l_2])
        L_cal[neuron_index] = layer.:γ[neuron_index] / sqrt(layer.:σ²[neuron_index] + layer.:ϵ^2) / 2
        C_cal = (- layer.:μ[neuron_index] * layer.:γ[neuron_index] / sqrt(layer.:σ²[neuron_index] + layer.:ϵ^2)) + layer.:β[neuron_index]
        Q = [zeros(n[l_2], n[l_2]) L_cal; L_cal' C_cal]

    elseif layer isa LayerNorm
        N = create_N_matrix(n[l_2])
        inv_W_hat = zeros(n[l_2])
        inv_W_hat[neuron_index] = pinv(layer.:diag.scale[neuron_index])
        b_hat = layer.:diag.bias[neuron_index]
        
        if all(N * inv_W_hat .== 0.0) || b_hat == 0 #Or is it N*inv_W_hat that should be checked
            Q = [zeros(n[l_2], n[l_2]) (N * inv_W_hat/2); (N * inv_W_hat/2)' b_hat]
            println("Flat")
        else
            Q_cal = (n[l_2] * N * inv_W_hat * inv_W_hat' * N) - ((inv_W_hat' * inv_W_hat)^2 * b_hat^2 * N)
            Q = [Q_cal zeros(n[l_2]); zeros(n[l_2])' 0]
            push!(Q_tilde, [zeros(n[l_2], n[l_2]) (N * inv_W_hat * b_hat); (N * inv_W_hat * b_hat)' 0])
            push!(layer_pattern, 0)
            println("Non-Flat")
        end
        Q_check, Q_m_check = CAB_step_layernorm(CAB_init_matrix(n[l_2], neuron_index), l_2)
    end

    Q = normalise_matrix(Q)
    for Qi in Q_tilde
        Qi = normalise_matrix(Qi)
    end

    partition_layer[0] = QuadraticPartitionEntry(layer_pattern, Q, Q_tilde, nothing, Vector{UInt128}(), "Boundary")
    CAB_partition_tree[1] = partition_layer

    for l in l_2-1:-1:1
        fm = init_fm(n[l]; epsilon = 1e-3)
        partition_layer = Dict{UInt128, QuadraticPartitionEntry}()
        partition_key = 0
        for (super_partition_key, super_partition) in CAB_partition_tree[l_2-l]
            # THIS DOES NOT CONSIDER LAYERS WITHOUT RELU !!!!!
            for layer_pattern in all_activation_patterns(n[l+1])
                Q_tilde = Vector{Matrix{Float64}}()
                if model[l] isa Dense
                    #Inverse ReLU function
                    P = Diagonal(layer_pattern)
                    Q_z = CAB_step_relu(super_partition.Q,P)

                    Q = CAB_step_dense(Q_z, l)
                    for i in 1:(n[l+1])
                        Li_cal = model[l].weight[i,:]/2
                        push!(Q_tilde, [zeros(n[l], n[l]) Li_cal; Li_cal' model[l].bias[i]])
                    end

                    for i in 1:length(super_partition.Q_tilde)
                        Qi_z = CAB_step_relu(super_partition.Q_tilde[i], P)
                        Qi = CAB_step_dense(Qi_z, l)
                        push!(Q_tilde, Qi)
                    end
                    
                    pattern = vcat(layer_pattern, super_partition.pattern)
                    nonvoid_bool, boundary_bool, tag = QP_feasibility(fm, Q, Q_tilde, pattern)

                    if boundary_bool
                        partition_layer[partition_key] = QuadraticPartitionEntry(pattern, Q, Q_tilde, super_partition_key, Vector{UInt128}(), tag)
                        push!(super_partition.sub_partitions, partition_key)
                        partition_key+=1
                    end
                elseif model[l] isa BatchNorm
                    #Inverse ReLU function
                    P = Diagonal(layer_pattern)
                    Q_z = CAB_step_relu(super_partition.Q,P)

                    Q = CAB_step_batchnorm(Q_z, l)

                    sigma = sqrt.(model[l].:σ² .+ model[l].:ϵ^2)
                    for i in 1:n[l]
                        Li_cal = zeros(n[l])
                        Li_cal[i] = model[l].:γ[i] / sigma[i] / 2
                        Ci_cal = (- model[l].:μ[i] * model[l].:γ[i]) + model[l].:β[i]
                        push!(Q_tilde, [zeros(n[l], n[l]) Li_cal; Li_cal' Ci_cal])
                    end

                    for i in 1:length(super_partition.Q_tilde)
                        Qi_z = CAB_step_relu(super_partition.Q_tilde[i], P)
                        Qi = CAB_step_batchnorm(Qi_z, l)
                        push!(Q_tilde, Qi)
                    end

                    pattern = vcat(layer_pattern, super_partition.pattern)
                    nonvoid_bool, boundary_bool, tag = QP_feasibility(fm, Q, Q_tilde, pattern)

                    if boundary_bool
                        partition_layer[partition_key] = QuadraticPartitionEntry(pattern, Q, Q_tilde, super_partition_key, Vector{UInt128}(), tag)
                        push!(super_partition.sub_partitions, partition_key)
                        partition_key+=1
                    end
                elseif model[l] isa LayerNorm
                    for mirror_pattern in all_activation_patterns(n[l+1])
                        empty!(Q_tilde)
                        #Inverse ReLU function
                        P = Diagonal(BitVector([mirror_pattern[i] ? (model[l].diag.bias[i] >= 0) : layer_pattern[i] for i in eachindex(layer_pattern)]))
                        Q_z = CAB_step_relu(super_partition.Q,P)

                        N = create_N_matrix(n[l])
                        pattern = BitVector()
                        inv_W_hat = pinv(2 * Q_z[1:end-1, end]' * Diagonal(model[l].:diag.scale))
                        b_hat = ((2 * Q_z[1:end-1, end]' * model[l].:diag.bias) + Q_z[end, end])
                        Q, Q_m = CAB_step_layernorm(Q_z, l)
                        push!(Q_tilde, Q_m)
                        push!(pattern, 0)
                        # New Q_tilde conditions
                        for i in 1:n[l]
                            Qi, Qi_m = CAB_step_layernorm(CAB_init_matrix(n[l], i), l)
                            push!(Q_tilde, Qi)
                            push!(pattern, layer_pattern[i])
                            push!(Q_tilde, Qi_m)
                            push!(pattern, mirror_pattern[i])
                        end

                        # Mapped Q_tilde conditions
                        for i in 1:length(super_partition.Q_tilde)
                            println("Passed Down Condition")
                            Qi_z = CAB_step_relu(super_partition.Q_tilde[i], P)
                            Qi, Qi_m = CAB_step_layernorm(Qi_z, l)
                            push!(Q_tilde, Qi)
                            push!(pattern, super_partition.pattern[i])
                        end

                        nonvoid_bool, boundary_bool, tag = QP_feasibility(fm, Q, Q_tilde, pattern)

                        if boundary_bool
                            partition_layer[partition_key] = QuadraticPartitionEntry(pattern, Q, Q_tilde, super_partition_key, Vector{UInt128}(), tag)
                            push!(super_partition.sub_partitions, partition_key)
                            partition_key+=1
                        end
                    end
                end
                #print(frame, l_2, neuron_index, l, vcat(layer_pattern, super_partition.pattern), Q)
                #println("Frame:", frame, " Neuron: ", l_2, ",", neuron_index, " CAB Layer: ", l, " Pattern: ", pattern, " Tag: ", tag)
            end
        end
        CAB_partition_tree[l_2 + 1 - l] = partition_layer
        empty!(fm.optimisation_model)
    end

    if to_save
        if isnothing(frame)
            save_path = "data/CAB_partition_tree.jlser"
        else
            save_path = @sprintf("data/CAB_partition_tree_%04d.jlser", frame)
        end
        println("Saving CAB tree (including non-boundary) to $save_path using Serialization")
        open(save_path, "w") do io
            serialize(io, CAB_partition_tree)
        end
    end
    return CAB_partition_tree
end

function calculate_quadratic_CAB_neuron_table(model::Flux.Chain, frame::Union{Int, Nothing} = nothing, to_save::Bool = true)
    n = get_n(model, 2)
    L = length(n) - 1
    partition_neuron_table = Vector{Vector{Dict{UInt128, QuadraticPartitionEntry}}}(undef, L)
    for l in 1:L
        partition_neuron_layer = Vector{Dict{UInt128, QuadraticPartitionEntry}}(undef, n[l+1])
        for i in 1:n[l+1]
            partition_neuron_layer[i] = calculate_quadratic_CAB_partition_tree(model, l, i, frame, false)[end]
        end
        partition_neuron_table[l] = partition_neuron_layer
    end
    if to_save
        if isnothing(frame)
            save_path = "data/partition_neuron_table.jlser"
        else
            save_path = @sprintf("data/partition_neuron_table_%04d.jlser", frame)
        end
        println("Saving CAB neuron table (including non-boundary) to $save_path using Serialization")
        open(save_path, "w") do io
            serialize(io, partition_neuron_table)
        end
    end
    return partition_neuron_table
end

function plot_quadratic_CAB_frame!(ax::Axis, outputs::Vector{Dict}, neuron_layer::Int, neuron_index::Int, partition_neuron_table:: Vector{Vector{Dict{UInt128, QuadraticPartitionEntry}}}, frame::Int)
    # --- Data Setup ---
    x, y = CAB_GRID.x, CAB_GRID.y
    points_aug = CAB_GRID.points_aug
    points_aug_T = CAB_GRID.points_aug_T
    grid_size = size(CAB_GRID.Xg) 

    # --- Clear Previous Plot ---
    empty!(ax.scene.plots)
    ax.title = isnothing(frame) ? latexstring("z^{[$neuron_layer]}_$neuron_index \\text{ with CAB}") : latexstring("z^{[$neuron_layer]}_{$neuron_index} \\text{ with CAB at frame $frame}")
    ax.xlabel = latexstring("a^{[0]}_1")
    ax.ylabel = latexstring("a^{[0]}_2")
    #println("Creating Figure for Neuron Layer: $neuron_layer, Neuron Index: $neuron_index, Frame: $frame")

    # === Plot Outputs === #
    colors = [get(ColorSchemes.turbid, i/(length(partition_neuron_table)-1)) for i in 0:(length(partition_neuron_table)-1)]
    
    z = get(outputs[neuron_layer], :z, outputs[neuron_layer][:a])[neuron_index, :]
    Zg = reshape(z, grid_size)
    max_Zg = maximum(abs.(Zg))
    
    heatmap!(ax, x, y, Zg; colormap = reverse(cgrad(ColorSchemes.broc)), colorrange = (-max_Zg, max_Zg), alpha = 1)

    # === Plot CAB === #
    for (i, partition_neuron_layer) in enumerate(partition_neuron_table[1:neuron_layer])
        for (j, partition_neuron) in enumerate(partition_neuron_layer)
            if i != neuron_layer || j == neuron_index
                for (_, partition) in partition_neuron
                    if partition.tag == "Boundary"
                        z = sum((points_aug * partition.Q) .* points_aug, dims=2)
                        z2d = reshape(z, length(x), length(y))
                        # Compute contour using fully qualified Contour.jl
                        cl = Contour.contour(x, y, z2d, 0.0)
                        for poly in cl.lines
                            # manually extract points
                            contour_points_list = poly.vertices
                            contour_points_aug = hcat(reduce(vcat, [collect(p)' for p in contour_points_list]), ones(length(contour_points_list)))
                            keep = trues(length(contour_points_list))

                            if !isempty(partition.Q_tilde)
                                orthant = 2 .* Int.(partition.pattern) .- 1
                                for condition_i in 1:length(partition.Q_tilde)
                                    Qc = Symmetric(partition.Q_tilde[condition_i])
                                    vals = orthant[condition_i] .* sum((contour_points_aug * Qc) .* contour_points_aug, dims=2)
                                    keep .= keep .& (vec(vals) .> -1e-2)
                                end
                            end

                            if any(keep)
                                thickness = i == neuron_layer ? 3 : 2
                                contour_x = getindex.(contour_points_list, 1)[keep]
                                contour_y = getindex.(contour_points_list, 2)[keep]
                                lines!(ax, contour_x, contour_y, color=colors[i], linewidth=thickness)
                            end
                        end
                    end
                end
            end
        end
    end
end

function plot_quadratic_CAB(models::Vector{Flux.Chain}, neuron_layer::Int, neuron_index::Int, frame::Union{Int, Nothing} = nothing, to_save::Bool = true)
    
    # --- Load File ---
    if isnothing(frame)
        load_path = "data/partition_neuron_table.jlser"
    else
        load_path = @sprintf("data/partition_neuron_table_%04d.jlser", frame)
    end
    partition_neuron_table = deserialize(load_path)
    
    # --- Figure Setup ---
    title_text = isnothing(frame) ? latexstring("z^{[$neuron_layer]}_$neuron_index \\text{ with CAB}") : latexstring("z^{[$neuron_layer]}_{$neuron_index} \\text{ with CAB at frame $frame}")
    fig = Figure(size = (900, 600))
    ax = Axis(fig[1, 1],
        title = title_text,
        xlabel = L"x_1",
        ylabel = L"x_2",
        aspect = DataAspect(),
        limits = ((-5, 5), (-5, 5))
    )

    # --- Call shared plotting logic ---
    outputs = get_z(models[frame+1], CAB_GRID.points_T)
    plot_quadratic_CAB_frame!(ax, outputs, neuron_layer, neuron_index, partition_neuron_table,frame)

    # --- Legend for Layers (recomputed here for fig) ---
    colors = [get(ColorSchemes.turbid, i/(length(models[frame+1])-1)) for i in 0:(length(models[frame+1])-1)]

    for (layer_idx, col) in enumerate(colors)
        label_idx = length(colors) - layer_idx + 1
        lines!(ax, [NaN], [NaN], color=col, linewidth=2, label="Layer $label_idx")
    end
    axislegend(ax, position=:lt)

    # --- Colorbars ---
    subgrid = fig[1, 2] = GridLayout()
    Colorbar(subgrid[1, 1], colormap = reverse(cgrad(ColorSchemes.bam)), limits = (-10, 10), label = "Boundary", width = 20, height = Relative(0.9))
    Colorbar(subgrid[1, 2], colormap = reverse(cgrad(ColorSchemes.broc)), limits = (-10, 10), label = "Non-Boundary", width = 20, height = Relative(0.9))

    # --- Save Figure ---
    if to_save
        if isnothing(frame)
            save_path = "CAB_plot.png"
        else
            save_path = @sprintf("plot_store/CAB_plot_%04d.png", frame)
        end
        println("Saving CAB plot to $save_path")
        save(save_path, fig)
    end

    return fig
end

function create_quadratic_animation(models::Vector{Flux.Chain}, total_frame::Int; output_path::String = "CAB_animation.mp4", framerate::Int = 10)
    n = get_n(models[1], 2)
    
    # === Figure ===
    fig = Figure(size = (1280, 720))
    rowgap!(fig.layout, 5)         # 5px vertical spacing
    colgap!(fig.layout, 5)         # 5px horizontal spacing

    # === Title ===
    Label(fig[0, 2], "Pre-activations and CAB's of all neurons", fontsize = 28, tellwidth = false, tellheight = true)
    rowsize!(fig.layout, 0, Auto(false))

    # === Legend for Layers ===
    colors = [get(ColorSchemes.turbid, i/(length(n)-2)) for i in 0:(length(n)-2)]
    dummy_axis = Axis(fig.scene)  # Create axis not added to layout
    legend_lines = [lines!(dummy_axis, [NaN], [NaN], color=col, linewidth=2) for col in colors]
    Legend(fig[1, 1], legend_lines, ["Layer $i" for i in 1:length(colors)]; title = "CAB Legend")
    colsize!(fig.layout, 1, Auto(false))

    # === Colorbars === 
    subgrid = fig[1, 3] = GridLayout()
    colgap!(subgrid, 2)
    Colorbar(subgrid[1, 1], colormap = reverse(cgrad(ColorSchemes.Reds)), limits = (-10, 10), label = "Boundary", width = 10, height = Relative(0.9), flip_vertical_label = true)
    Colorbar(subgrid[1, 2], colormap = reverse(cgrad(ColorSchemes.Blues)), limits = (-10, 10), label = "Non-Boundary", width = 10, height = Relative(0.9), flip_vertical_label = true)

    colsize!(subgrid, 1, Auto(false))
    colsize!(subgrid, 2, Auto(false))
    colsize!(fig.layout, 3, Auto(false))

    # === Plots ===
    axes_grid = Vector{Vector{Axis}}(undef, length(n)-1)
    for row in 1:length(n)-1
        neuron_layer = length(n) - row
        
        row_grid = fig[row, 2] = GridLayout()  
        colgap!(row_grid, 5)
        rowsize!(fig.layout, row, Auto(false))

        axes_grid[row] = Vector{Axis}(undef, n[neuron_layer + 1])
        for neuron_index in 1:n[neuron_layer+1]
            axes_grid[row][neuron_index] = Axis(row_grid[1, neuron_index], aspect = DataAspect())
            colsize!(row_grid, neuron_index, Auto(false))
        end
    end
    rowsize!(fig.layout, 1, Relative(0.4))
    colsize!(fig.layout, 2, Relative(0.8))

    # === Animation Recording ===
    record(fig, output_path, 0:total_frame; framerate = framerate) do frame
        partition_neuron_table = deserialize(@sprintf("data/partition_neuron_table_%04d.jlser", frame))
        outputs = get_z(models[frame+1], CAB_GRID.points_T)
        for row in 1:length(n)-1
            neuron_layer = length(n) - row
            for neuron_index in 1:n[neuron_layer+1]
                ax = axes_grid[row][neuron_index]
                empty!(ax)
                plot_quadratic_CAB_frame!(ax, outputs, neuron_layer, neuron_index, partition_neuron_table, frame)
            end
        end
    end

    println("Animation saved to $output_path") # Note that things are saved live in the previous block
end

end # module