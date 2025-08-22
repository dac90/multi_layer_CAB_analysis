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

# Include other files
include("CAB_visualiser_empirical.jl")
using .CAB_visualiser_empirical
export create_empirical_animation, plot_empirical_CAB, plot_empirical_CAB_3D

# === Export Functions === #
export calculate_CAB_partition_tree, calculate_CAB_boundary_tree, get_CAB_partition_tree, calculate_CAB_neuron_table, get_CAB_neuron_table, plot_CAB, create_animation, create_animation_2, get_partition_count, plot_partition_count, plot_partition_count_quantiles

# === Pre-Defined Variables === #
cmap_boundary = reverse(cgrad(ColorSchemes.Reds, 256))
cmap_null     = reverse(cgrad(ColorSchemes.Purples, 256))
cmap_nonbound = reverse(cgrad(ColorSchemes.Blues, 256))

# === Feasability Model Structure === #
mutable struct FeasibilityModel
    linear_model::Model
    x::Vector{VariableRef}
    epsilon::Float64
end

# === Partition Data Structure === #
struct PartitionEntry
    phi::Vector{Float64}
    pattern::Vector{BitVector}
    W_hat::Matrix{Float64}
    b_hat::Vector{Float64}
    W_tilde::Matrix{Float64}
    b_tilde::Vector{Float64}
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
    points_T::Matrix{Float64}
end

const CAB_GRID = let
    x = LinRange(-5, 5, 100)
    y = LinRange(-5, 5, 100)
    Xg = repeat(reshape(x, :, 1), 1, length(y))
    Yg = repeat(reshape(y, 1, :), length(x), 1)
    points = hcat(vec(Xg), vec(Yg))
    points_T = points'
    ConstGrid(x, y, Xg, Yg, points, points_T)
end

"""
    all_activation_patterns(sizes::Vector{Int}) -> Iterator{Vector{BitVector}}

Generates all possible ReLU activation patterns for each layer size.
Returns an iterator of activation patterns as Vector{BitVector}.
"""
function all_activation_patterns(layer_sizes::Vector{Int})
    function binary_patterns(n::Int) # Returns a Vector{Bitvectors}, containing all possible length n binary combinations
        m = 2^n
        patterns = Vector{BitVector}(undef, m)
        for i in 0:m-1
            bv = falses(n)
            for j in 1:n
                bv[j] = (i >> (n-j)) & 1 == 1
            end
            patterns[i+1] = bv
        end
        return patterns
    end

    pattern_lists = [binary_patterns(n) for n in layer_sizes] # A list of Vector{Bitvectors} of activation patterns at each layer
    return (collect(p) for p in Iterators.product(pattern_lists...)) # The the options at each layer are combined, and turned into an iterator
end

"""
    unwrap_network(pattern::Vector{BitVector}, frame) -> W_hat::Matrix, b_hat::Vector, W_tilde:Matrix, b_tilde::Matrix

Given the model and a relu activation pattern, computes the affine function `f(x) = W_hat * x + b_hat
for that region, as well as W_tilde and b_tilde, which output hidden layer pre-activations.
"""
function unwrap_network(model::Flux.Chain, pattern::Vector{BitVector})

    L = size(pattern)[1] + 1  # Total layers

    # Initialize W_hat and b_hat
    W_hat = model[1].weight
    b_hat = model[1].bias

    # Sequences (store all intermediate W_hat and b_hat)
    W_hat_seq = [W_hat]
    b_hat_seq = [b_hat]

    # Loop through layers
    for l in 1:L-1
        D = Diagonal(Float64.(pattern[l]))
        W = model[l+1].weight
        b = model[l+1].bias

        W_hat = W * D * W_hat
        b_hat = W * D * b_hat .+ b

        # Append current W_hat and b_hat to the sequences
        push!(W_hat_seq, W_hat)
        push!(b_hat_seq, b_hat)
    end

    # This matrix and vector output the result in all hidden layers of the matrix, but not the actual output.
    W_tilde = reduce(vcat, W_hat_seq[1:end-1])
    b_tilde = reduce(vcat, b_hat_seq[1:end-1])

    return W_hat, b_hat, W_tilde, b_tilde
end

"""
    calculate_CAB_partition_tree(layer_sizes::Vector{Int}, neuron_index::Int, frame::Union{Int, Nothing} = nothing)

Calculates the CAB of a Neuron in all lower layers, latent and otherwise. Ignores void partitions effectively
"""

function calculate_CAB_partition_tree(model::Flux.Chain, layer_sizes::Vector{Int}, neuron_index::Int, frame::Union{Int, Nothing} = nothing, to_save::Bool = true)

    function init_fm(x_size::Int; solver=HiGHS.Optimizer, epsilon=1e-6)
        linear_model = Model(solver)
        set_silent(linear_model)
        @variable(linear_model, x[1:x_size])
        return FeasibilityModel(linear_model, x, epsilon)
    end

    function LP_feasability(fm::FeasibilityModel, A::Matrix{Float64}, b::Vector{Float64}, pattern::Vector{BitVector})
        # Input variables
        orthant = 2 .* Int.(collect(Iterators.flatten(pattern))) .- 1
        A_flipped = Diagonal(orthant) * A
        b_flipped = b .* orthant

        # Add constraints: W_scaled * x + b_scaled >= epsilon (small positive)
        epsilon = 1e-6
        con_refs = @constraint(fm.linear_model, A_flipped * fm.x .+ b_flipped .>= epsilon)

        optimize!(fm.linear_model)
        status = termination_status(fm.linear_model)

        # Delete constraints for next iteration
        foreach(c -> delete(fm.linear_model, c), con_refs)
        return status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
    end

    L = size(layer_sizes)[1] - 1
    CAB_partition_tree = Vector{Dict{UInt128, PartitionEntry}}(undef, L)
    partition_layer = Dict{UInt128, PartitionEntry}()

    W_init = model[L].weight[neuron_index:neuron_index, :]
    b_init = model[L].bias[neuron_index:neuron_index]
    phi_init = -pinv(W_init) * b_init
    partition_layer[0] = PartitionEntry(phi_init, Vector{BitVector}(), W_init, b_init, Matrix{Float64}(undef, 0, size(W_init)[2]), Vector{Float64}(), nothing, Vector{UInt128}(), "Boundary")
    CAB_partition_tree[1] = partition_layer

    for l in L-2:-1:0
        fm1 = init_fm(layer_sizes[l+1])
        fm2 = init_fm(layer_sizes[l+1]-1)
        partition_layer = Dict{UInt128, PartitionEntry}()
        for (super_partition_key, super_partition) in CAB_partition_tree[L-l-1]
            for layer_pattern in all_activation_patterns(layer_sizes[l+2:l+2])
                pattern =  vcat(layer_pattern, super_partition.pattern)

                D = Diagonal(layer_pattern[1])
                W = model[l+1].weight
                b = model[l+1].bias

                W_hat = super_partition.W_hat * D * W
                b_hat = super_partition.b_hat + (super_partition.W_hat * D * b)
                W_tilde = vcat(W, super_partition.W_tilde * D * W)
                b_tilde = vcat(b, super_partition.b_tilde + (super_partition.W_tilde * D * b))

                phi = -pinv(W_hat) * b_hat
                nonvoid_bool = LP_feasability(fm1, W_tilde, b_tilde, pattern)

                tag = "N/A"
                if nonvoid_bool
                    if all(phi .== 0.0)
                        tag = "Null"
                    else    # If non-void and non-null, perform a projection from the space of the CAB plane and repeat LP test to test for boundary region
                        Q, _ = qr([phi I])
                        phi_ortho = Q[:, 2:end]
                        W_tilde_proj = W_tilde * phi_ortho
                        b_tilde_proj = (W_tilde * phi) + b_tilde
                        boundary_bool = LP_feasability(fm2, W_tilde_proj, b_tilde_proj, pattern)
                        if boundary_bool
                            tag = "Boundary"
                        else
                            tag = "Non-Boundary"
                        end
                    end
                    partition_key = partition_key = foldl((acc, b) -> (acc << 1) | b, Iterators.flatten(( [1], collect(Iterators.flatten(pattern)) )), init = UInt128(0))
                    partition_layer[partition_key] = PartitionEntry(phi, pattern, W_hat, b_hat, W_tilde, b_tilde, super_partition_key, Vector{UInt128}(), tag)
                    push!(super_partition.sub_partitions, partition_key)
                end
            end
        end
        CAB_partition_tree[L - l] = partition_layer
        empty!(fm1.linear_model)
        empty!(fm2.linear_model)
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
    calculate_CAB_boundary_tree(layer_sizes::Vector{Int}, neuron_index::Int, frame::Union{Int, Nothing} = nothing)

Calculates the CAB of a Neuron in all lower layers, latent and otherwise. Ignores void and non-boundary partitions effectively
"""

function calculate_CAB_boundary_tree(model::Flux.Chain, layer_sizes::Vector{Int}, neuron_index::Int, frame::Union{Int, Nothing} = nothing, to_save::Bool=true)

    function init_fm(x_size::Int; solver=HiGHS.Optimizer, epsilon=1e-6)
        linear_model = Model(solver)
        set_silent(linear_model)
        @variable(linear_model, x[1:x_size])
        return FeasibilityModel(linear_model, x, epsilon)
    end

    function LP_feasability(fm::FeasibilityModel, A::Matrix{Float64}, b::Vector{Float64}, pattern::Vector{BitVector})
        # Input variables
        orthant = 2 .* Int.(collect(Iterators.flatten(pattern))) .- 1
        A_flipped = Diagonal(orthant) * A
        b_flipped = b .* orthant

        # Add constraints: W_scaled * x + b_scaled >= epsilon (small positive)
        epsilon = 1e-6
        con_refs = @constraint(fm.linear_model, A_flipped * fm.x .+ b_flipped .>= epsilon)

        optimize!(fm.linear_model)
        status = termination_status(fm.linear_model)

        # Delete constraints for next iteration
        foreach(c -> delete(fm.linear_model, c), con_refs)
        return status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
    end

    L = size(layer_sizes)[1] - 1
    CAB_partition_tree = Vector{Dict{UInt128, PartitionEntry}}(undef, L)
    partition_layer = Dict{UInt128, PartitionEntry}()
    
    W_init = model[L].weight[neuron_index:neuron_index, :]
    b_init = model[L].bias[neuron_index:neuron_index]
    phi_init = -pinv(W_init) * b_init
    partition_layer[0] = PartitionEntry(phi_init, Vector{BitVector}(), W_init, b_init, Matrix{Float64}(undef, 0, size(W_init)[2]), Vector{Float64}(), nothing, Vector{UInt128}(0), "Boundary")
    CAB_partition_tree[1] = partition_layer

    for l in L-2:-1:0
        fm = init_fm(layer_sizes[l+1]-1)
        partition_layer = Dict{UInt128, PartitionEntry}()
        for (_, super_partition) in CAB_partition_tree[L-l-1]
            for layer_pattern in all_activation_patterns(layer_sizes[l+2:l+2])
                pattern =  vcat(layer_pattern, super_partition.pattern)

                D = Diagonal(layer_pattern[1])
                W = model[l+1].weight
                b = model[l+1].bias

                W_hat = super_partition.W_hat * D * W
                b_hat = super_partition.b_hat + (super_partition.W_hat * D * b)
                W_tilde = vcat(W, super_partition.W_tilde * D * W)
                b_tilde = vcat(b, super_partition.b_tilde + (super_partition.W_tilde * D * b))

                phi = -pinv(W_hat) * b_hat
                if !all(phi .== 0.0) # If non-void and non-null, perform a projection from the space of the CAB plane and repeat LP test to test for boundary region
                    Q, _ = qr([phi I])
                    phi_ortho = Q[:, 2:end]
                    W_tilde_proj = W_tilde * phi_ortho
                    b_tilde_proj = (W_tilde * phi) + b_tilde
                    boundary_bool = LP_feasability(fm, W_tilde_proj, b_tilde_proj, pattern)
                    if boundary_bool
                        partition_key = foldl((acc, b) -> (acc << 1) | b, Iterators.flatten(( [1], collect(Iterators.flatten(pattern)) )), init = UInt128(0))
                        partition_layer[partition_key] = PartitionEntry(phi, pattern, W_hat, b_hat, W_tilde, b_tilde, super_partition_key, Vector{UInt128}(), tag)
                        push!(super_partition.sub_partitions, partition_key)
                    end
                end
            end
        end
        CAB_partition_tree[L - l] = partition_layer
        empty!(fm.linear_model)
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

function graph_CAB_partition_tree!(ax::Axis, frame::Int; show_labels::Bool)
    # --- Load data ---
    load_path = @sprintf("data/CAB_partition_tree_%04d.jlser", frame)
    CAB_partition_tree = deserialize(load_path)

    # --- Flatten all layers into one dict ---
    all_parts = Dict{UInt128, PartitionEntry}()
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

function calculate_CAB_neuron_table(model::Flux.Chain, layer_sizes::Vector{Int}, frame::Union{Int, Nothing} = nothing, to_save::Bool = true)
    L = size(layer_sizes)[1] - 1
    partition_neuron_table = Vector{Vector{Dict{UInt128, PartitionEntry}}}(undef, L)
    for l in 1:L
        partition_neuron_layer = Vector{Dict{UInt128, PartitionEntry}}(undef, layer_sizes[l+1])
        for i in 1:layer_sizes[l+1]
            partition_neuron_layer[i] = calculate_CAB_partition_tree(model, layer_sizes[1:l+1], i, frame, false)[end]
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

function plot_CAB_frame!(ax::Axis, neuron_layer::Int, neuron_index::Int, partition_neuron_table:: Vector{Vector{Dict{UInt128, PartitionEntry}}}, frame::Int)
    # --- Data Setup ---
    x, y = CAB_GRID.x, CAB_GRID.y
    points = CAB_GRID.points
    points_T = CAB_GRID.points_T
    grid_size = size(CAB_GRID.Xg) 

    mask = falses(size(points, 1))
    z = Vector{Float64}(undef, size(points, 1))

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
        if isempty(partition.pattern) || isempty(partition.W_tilde) || isempty(partition.b_tilde)
            mask .= true
        else
            orthant = 2 .* Int.(reduce(vcat, partition.pattern)) .- 1
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
    heatmap!(ax, x, y, Zg_boundary; colormap = reverse(cgrad(ColorSchemes.Reds)), colorrange = color_limits, alpha=1)
    heatmap!(ax, x, y, Zg_null;     colormap = reverse(cgrad(ColorSchemes.Purples)), colorrange = color_limits, alpha=1)
    heatmap!(ax, x, y, Zg_non_boundary; colormap = reverse(cgrad(ColorSchemes.Blues)), colorrange = color_limits, alpha=1)

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
                            orthant = 2 .* Int.(reduce(vcat, partition.pattern)) .- 1
                            W_tilde_flipped = Diagonal(orthant) * partition.W_tilde
                            b_tilde_flipped = partition.b_tilde .* orthant
                            CAB_points_mask .&= vec(all(W_tilde_flipped * CAB_points' .+ b_tilde_flipped .> 0, dims = 1))
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
    plot_CAB(weights::Dict, layer_sizes::Vector{Int})

Plots all activation boundaries for a 2D input network.
"""

function plot_CAB(layer_sizes::Vector{Int}, neuron_layer::Int, neuron_index::Int, frame::Union{Int, Nothing} = nothing, to_save::Bool = true)
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
    colors = [get(ColorSchemes.turbid, i/(length(layer_sizes)-2)) for i in 0:(length(layer_sizes)-2)]

    for (layer_idx, col) in enumerate(colors)
        label_idx = length(colors) - layer_idx + 1
        lines!(ax, [NaN], [NaN], color=col, linewidth=2, label="Layer $label_idx")
    end
    axislegend(ax, position=:lt)

    # --- Colorbars ---
    subgrid = fig[1, 2] = GridLayout()
    Colorbar(subgrid[1, 1], colormap = reverse(cgrad(ColorSchemes.Reds)), limits = (-10, 10), label = "Boundary", width = 20, height = Relative(0.9))
    Colorbar(subgrid[1, 2], colormap = reverse(cgrad(ColorSchemes.Purples)), limits = (-10, 10), label = "Null", width = 20, height = Relative(0.9))
    Colorbar(subgrid[1, 3], colormap = reverse(cgrad(ColorSchemes.Blues)), limits = (-10, 10), label = "Non-Boundary", width = 20, height = Relative(0.9))

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

function create_animation(layer_sizes::Vector{Int}, total_frame::Int; output_path::String = "CAB_animation.mp4", framerate::Int = 10)
    # === Figure ===
    fig = Figure(size = (1280, 720))
    rowgap!(fig.layout, 5)         # 5px vertical spacing
    colgap!(fig.layout, 5)         # 5px horizontal spacing

    # === Title ===
    Label(fig[0, 2], "Pre-activations and CAB's of all neurons", fontsize = 28, tellwidth = false, tellheight = true)
    rowsize!(fig.layout, 0, Auto(false))

    # === Legend for Layers ===
    colors = [get(ColorSchemes.turbid, i/(length(layer_sizes)-2)) for i in 0:(length(layer_sizes)-2)]
    dummy_axis = Axis(fig.scene)  # Create axis not added to layout
    legend_lines = [lines!(dummy_axis, [NaN], [NaN], color=col, linewidth=2) for col in colors]
    Legend(fig[1, 1], legend_lines, ["Layer $i" for i in 1:length(colors)]; title = "CAB Legend")
    colsize!(fig.layout, 1, Auto(false))

    # === Colorbars === 
    subgrid = fig[1, 3] = GridLayout()
    colgap!(subgrid, 2)
    Colorbar(subgrid[1, 1], colormap = reverse(cgrad(ColorSchemes.Reds)), limits = (-10, 10), label = "Boundary", width = 10, height = Relative(0.9), flip_vertical_label = true)
    Colorbar(subgrid[1, 2], colormap = reverse(cgrad(ColorSchemes.Purples)), limits = (-10, 10), label = "Null", width = 10, height = Relative(0.9), flip_vertical_label = true)
    Colorbar(subgrid[1, 3], colormap = reverse(cgrad(ColorSchemes.Blues)), limits = (-10, 10), label = "Non-Boundary", width = 10, height = Relative(0.9), flip_vertical_label = true)

    colsize!(subgrid, 1, Auto(false))
    colsize!(subgrid, 2, Auto(false))
    colsize!(subgrid, 3, Auto(false))
    colsize!(fig.layout, 3, Auto(false))

    # === Plots ===
    axes_grid = Vector{Vector{Axis}}(undef, length(layer_sizes)-1)
    for row in 1:length(layer_sizes)-1
        neuron_layer = length(layer_sizes) - row
        
        row_grid = fig[row, 2] = GridLayout()  
        colgap!(row_grid, 5)
        rowsize!(fig.layout, row, Auto(false))

        axes_grid[row] = Vector{Axis}(undef, layer_sizes[neuron_layer + 1])
        for neuron_index in 1:layer_sizes[neuron_layer+1]
            axes_grid[row][neuron_index] = Axis(row_grid[1, neuron_index], aspect = DataAspect())
            colsize!(row_grid, neuron_index, Auto(false))
        end
    end
    rowsize!(fig.layout, 1, Relative(0.4))
    colsize!(fig.layout, 2, Relative(0.8))

    # === Animation Recording ===
    record(fig, output_path, 0:total_frame; framerate = framerate) do frame
        partition_neuron_table = deserialize(@sprintf("data/partition_neuron_table_%04d.jlser", frame))
        for row in 1:length(layer_sizes)-1
            neuron_layer = length(layer_sizes) - row
            for neuron_index in 1:layer_sizes[neuron_layer+1]
                ax = axes_grid[row][neuron_index]
                empty!(ax)
                plot_CAB_frame!(ax, neuron_layer, neuron_index, partition_neuron_table, frame)
            end
        end
    end

    println("Animation saved to $output_path") # Note that things are saved live in the previous block
end

function create_animation_2(layer_sizes::Vector{Int}, total_frame::Int; output_path::String = "CAB_animation_2.mp4", framerate::Int = 10)
    # === Figure ===
    fig = Figure(size = (1280, 720))
    rowgap!(fig.layout, 5)         # 5px vertical spacing
    colgap!(fig.layout, 5)         # 5px horizontal spacing

    # === Title ===
    Label(fig[0, 2], "Pre-activations and CAB's of all neurons", fontsize = 28, tellwidth = false, tellheight = true)
    rowsize!(fig.layout, 0, Auto(false))

    # === Legend for Layers ===
    colors = [get(ColorSchemes.turbid, i/(length(layer_sizes)-2)) for i in 0:(length(layer_sizes)-2)]
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
    Colorbar(subgrid[1, 1], colormap = reverse(cgrad(ColorSchemes.Reds)), limits = (-10, 10), label = "Boundary", width = 10, height = Relative(0.9), flip_vertical_label = true)
    Colorbar(subgrid[1, 2], colormap = reverse(cgrad(ColorSchemes.Purples)), limits = (-10, 10), label = "Null", width = 10, height = Relative(0.9), flip_vertical_label = true)
    Colorbar(subgrid[1, 3], colormap = reverse(cgrad(ColorSchemes.Blues)), limits = (-10, 10), label = "Non-Boundary", width = 10, height = Relative(0.9), flip_vertical_label = true)

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
        plot_CAB_frame!(ax_CAB, length(layer_sizes)-1, 1, partition_neuron_table, frame)
        empty!(ax_tree)
        graph_CAB_partition_tree!(ax_tree, frame; show_labels = false)
    end

    println("Animation saved to $output_path") # Note that things are saved live in the previous block
end

function get_partition_count(model::Flux.Chain, layer_sizes::Vector{Int}, neuron_index::Int)
    CAB_partition_tree = calculate_CAB_partition_tree(model, layer_sizes, neuron_index, nothing, false)
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

end # module