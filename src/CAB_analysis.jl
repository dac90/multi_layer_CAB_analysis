module CAB_analysis

# === Import Packages ===
using PyCall
using Serialization
using LinearAlgebra
using JuMP
using HiGHS
using GLMakie
using Colors
using ColorSchemes
using LaTeXStrings
using Printf
using ColorTypes
using ColorVectorSpace
using ImageCore
using FFMPEG

# === Export Functions === #
export save_params, get_params, calculate_partition_simple, get_partition_simple, calculate_partition_tree, calculate_boundary_tree, get_partition_tree, calculate_partition_all, calculate_boundary_all, get_partition_all, plot_CAB_all, create_animation, plot_partition_count

# === Pre-Defined Variables === #
cmap_boundary = reverse(cgrad(ColorSchemes.Reds, 256))
cmap_null     = reverse(cgrad(ColorSchemes.Purples, 256))
cmap_nonbound = reverse(cgrad(ColorSchemes.Blues, 256))

# === Feasability Model Structure === #
mutable struct FeasibilityModel
    model::Model
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
    tag::String
end

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

# === Save Network Parameters === #
"""
    save_params(model::PyObject, epoch::Union{Int, Nothing}::Union{Int, Nothing} = nothing, to_save = true) -> params::Dict{String, Any}

Given a PyTorch model (e.g. `torch.nn.Sequential`), save weight matrices and bias vectors to a '.jlser' file at 'path' with optional epoch, if to_save is true 
"""
function save_params(model::PyObject, epoch::Union{Int, Nothing} = nothing, to_save = true)
    torch = pyimport("torch")
    params = Dict{String, Any}()

    # --- Collect Parameters ---
    layer_index = 1
    for layer in model
        if pyisinstance(layer, torch.nn.Linear)
            params["W_$layer_index"] = Array(layer.weight.detach().numpy()) # Save weight matrix
            params["b_$layer_index"] = Array(layer.bias.detach().numpy()) # Save bias vector
            layer_index += 1
        end
    end

    # --- Save File ---
    if to_save
        if isnothing(epoch)
            save_path = "data/params.jlser"
        else
            save_path = @sprintf("data/params_%04d.jlser", epoch)
        end
        println("Saving parameters to $save_path using Serialization")
        open(save_path, "w") do io
            serialize(io, params)
        end
    end
    return params
end

# === Get Network Parameters ===
"""
    get_params(epoch) -> params::Dict{String, Any}

Deserializes the `.jlser` file at `path`, prints each entry's key, shape, and element type, and returns the dictionary for use in Python.
"""
function get_params(epoch::Union{Int, Nothing} = nothing)
    # --- Load File ---
    if isnothing(epoch)
        load_path = "data/params.jlser"
    else
        load_path = @sprintf("data/params_%04d.jlser", epoch)
    end
    params = deserialize(load_path)

    # --- Output Contents ---
    for k in sort(collect(keys(params))) # Sort to prevent random ordering
        v = params[k]
        println("$k => value: ", v)
    end

    return params
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
    unwrap_network(pattern::Vector{BitVector}, epoch) -> W_hat::Matrix, b_hat::Vector, W_tilde:Matrix, b_tilde::Matrix

Given params and a relu activation pattern, computes the affine function `f(x) = W_hat * x + b_hat
for that region, as well as W_tilde and b_tilde, which output hidden layer pre-activations.
"""
function unwrap_network(pattern::Vector{BitVector}, epoch::Union{Int, Nothing} = nothing)
    if isnothing(epoch)
        load_path = "data/params.jlser"
    else
        load_path = @sprintf("data/params_%04d.jlser", epoch)
    end
    params = deserialize(load_path)

    L = size(pattern)[1] + 1  # Total layers

    # Initialize W_hat and b_hat
    W_hat = params["W_1"]
    b_hat = params["b_1"]

    # Sequences (store all intermediate W_hat and b_hat)
    W_hat_seq = [W_hat]
    b_hat_seq = [b_hat]

    # Loop through layers
    for l in 1:L-1
        D = Diagonal(Float64.(pattern[l]))
        W = params["W_$(l+1)"]
        b = params["b_$(l+1)"]

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
    calculate_partition_simple(layer_sizes::Vector{Int}, epoch::Union{Int, Nothing} = nothing, to_save::Bool = true) -> partitions::Dict{UInt128, PartitionEntry}()

Calculates the CAB position vector in each partition.
Performs a feasability test using linear programming to determine whether a partition is void.
Subsequently performs a projection and another feasability test to determine whether a partition contains a boundary or not.
Saves all results as a in a FeasibilityModel struct in the '.jlser' file.
"""

function calculate_partition_simple(layer_sizes::Vector{Int}, epoch::Union{Int, Nothing} = nothing, to_save::Bool = true)

    function init_fm(x_size::Int; solver=HiGHS.Optimizer, epsilon=1e-6)
        model = Model(solver)
        set_silent(model)
        @variable(model, x[1:x_size])
        return FeasibilityModel(model, x, epsilon)
    end

    fm1 = init_fm(layer_sizes[1]) # LP Model for testing non-void partitions
    fm2 = init_fm(layer_sizes[1]-1) # LP Model for testing boundary partitions

    # === Feasability Model ===
    function LP_feasability(fm::FeasibilityModel, A::Matrix{Float64}, b::Vector{Float64}, pattern::Vector{BitVector})
        # --- Orthant to be tested ---
        orthant = orthant = 2 .* Int.(collect(Iterators.flatten(pattern))) .- 1
        A_flipped = Diagonal(orthant) * A
        b_flipped = b .* orthant
        
        # --- Add Constraints ---
        epsilon = 1e-6
        con_refs = @constraint(fm.model, A_flipped * fm.x .+ b_flipped .>= epsilon)

        # --- Test feasibility ---
        optimize!(fm.model)
        status = termination_status(fm.model)
        foreach(c -> delete(fm.model, c), con_refs) # Clear Constraints
        return status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED # Return Boolean
    end

    partitions = Dict{UInt128, PartitionEntry}()

    for pattern in all_activation_patterns(layer_sizes[2:end-1])
        W_hat, b_hat, W_tilde, b_tilde = unwrap_network(pattern, epoch)
        phi = -pinv(W_hat) * b_hat
        nonvoid_bool = LP_feasability(fm1, W_tilde, b_tilde, pattern)
        tag = "N/A"
        if nonvoid_bool
            if phi == [0.0, 0.0]
                tag = "Null"
            else    # If non-void and non-null, perform a projection from the space of the CAB plane and repeat LP test to test for boundary region
                # --- Test for Boundary Partition ---
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
            partitions[foldl((acc, b) -> (acc << 1) | b, Iterators.flatten(pattern); init=UInt128(0))] = PartitionEntry(phi, pattern, W_hat, b_hat, W_tilde, b_tilde, tag)
        end
    end

    if to_save
        if isnothing(epoch)
            save_path = "data/partitions.jlser"
        else
            save_path = @sprintf("data/partitions_%04d.jlser", epoch)
        end
        println("Saving partitions to $save_path using Serialization")
        open(save_path, "w") do io
            serialize(io, partitions)
        end
    end
    return partitions
end

"""
    get_partition_simple(path::String)

Deserializes the `.jlser` file at `path`, prints each partition's CAB position vector and tag
"""
function get_partition_simple(epoch)
    load_path = @sprintf("data/partitions_%04d.jlser", epoch)
    partitions = deserialize(load_path)

    for k in sort(collect(keys(partitions)))
        v = partitions[k]
        println("$k => phi: ", v.phi, ", tag: ", v.tag)
    end

    return partitions
end

"""
    calculate_partition_tree(layer_sizes::Vector{Int}, neuron_index::Int, epoch::Union{Int, Nothing} = nothing)

Calculates the CAB of a Neuron in all lower layers, latent and otherwise. Ignores void partitions effectively
"""

function calculate_partition_tree(layer_sizes::Vector{Int}, neuron_index::Int, epoch::Union{Int, Nothing} = nothing, to_save::Bool = true)

    function init_fm(x_size::Int; solver=HiGHS.Optimizer, epsilon=1e-6)
        model = Model(solver)
        set_silent(model)
        @variable(model, x[1:x_size])
        return FeasibilityModel(model, x, epsilon)
    end

    function LP_feasability(fm::FeasibilityModel, A::Matrix{Float64}, b::Vector{Float64}, pattern::Vector{BitVector})
        # Input variables
        orthant = 2 .* Int.(collect(Iterators.flatten(pattern))) .- 1
        A_flipped = Diagonal(orthant) * A
        b_flipped = b .* orthant

        # Add constraints: W_scaled * x + b_scaled >= epsilon (small positive)
        epsilon = 1e-6
        con_refs = @constraint(fm.model, A_flipped * fm.x .+ b_flipped .>= epsilon)

        optimize!(fm.model)
        status = termination_status(fm.model)

        # Delete constraints for next iteration
        foreach(c -> delete(fm.model, c), con_refs)
        return status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
    end

    if isnothing(epoch)
        load_path = "data/params.jlser"
    else
        load_path = @sprintf("data/params_%04d.jlser", epoch)
    end
    params = deserialize(load_path)

    L = size(layer_sizes)[1] - 1
    partition_tree = Vector{Dict{UInt128, PartitionEntry}}(undef, L)
    partition_layer = Dict{UInt128, PartitionEntry}()

    W_init = params["W_$(L)"][neuron_index:neuron_index, :]
    b_init = params["b_$(L)"][neuron_index:neuron_index]
    phi_init = -pinv(W_init) * b_init
    partition_layer[0] = PartitionEntry(phi_init, Vector{BitVector}(), W_init, b_init, Matrix{Float64}(undef, 0, size(W_init)[2]), Vector{Float64}(), "Boundary")
    partition_tree[1] = partition_layer

    for l in L-2:-1:0
        fm1 = init_fm(size(params["W_$(l+1)"])[2])
        fm2 = init_fm(size(params["W_$(l+1)"])[2]-1)
        partition_layer = Dict{UInt128, PartitionEntry}()
        for (_, super_partition) in partition_tree[L-l-1]
            for layer_pattern in all_activation_patterns(layer_sizes[l+2:l+2])
                pattern =  vcat(layer_pattern, super_partition.pattern)

                D = Diagonal(layer_pattern[1])
                W = params["W_$(l+1)"]
                b = params["b_$(l+1)"]

                W_hat = super_partition.W_hat * D * W
                b_hat = super_partition.b_hat + (super_partition.W_hat * D * b)
                W_tilde = vcat(W, super_partition.W_tilde * D * W)
                b_tilde = vcat(b, super_partition.b_tilde + (super_partition.W_tilde * D * b))

                phi = -pinv(W_hat) * b_hat
                nonvoid_bool = LP_feasability(fm1, W_tilde, b_tilde, pattern)

                tag = "N/A"
                if nonvoid_bool
                    if phi == [0.0, 0.0]
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
                    partition_layer[foldl((acc, b) -> (acc << 1) | b, Iterators.flatten(pattern); init=UInt128(0))] = PartitionEntry(phi, pattern, W_hat, b_hat, W_tilde, b_tilde, tag)
                end
            end
        end
        partition_tree[L - l] = partition_layer
        empty!(fm1.model)
        empty!(fm2.model)
    end
    if to_save
        if isnothing(epoch)
            save_path = "data/partition_tree.jlser"
        else
            save_path = @sprintf("data/partition_tree_%04d.jlser", epoch)
        end
        println("Saving CAB tree (including non-boundary) to $save_path using Serialization")
        open(save_path, "w") do io
            serialize(io, partition_tree)
        end
    end
    return partition_tree
end

"""
    calculate_boundary_tree(layer_sizes::Vector{Int}, neuron_index::Int, epoch::Union{Int, Nothing} = nothing)

Calculates the CAB of a Neuron in all lower layers, latent and otherwise. Ignores void and non-boundary partitions effectively
"""

function calculate_boundary_tree(layer_sizes::Vector{Int}, neuron_index::Int, epoch::Union{Int, Nothing} = nothing, to_save::Bool=true)

    function init_fm(x_size::Int; solver=HiGHS.Optimizer, epsilon=1e-6)
        model = Model(solver)
        set_silent(model)
        @variable(model, x[1:x_size])
        return FeasibilityModel(model, x, epsilon)
    end

    function LP_feasability(fm::FeasibilityModel, A::Matrix{Float64}, b::Vector{Float64}, pattern::Vector{BitVector})
        # Input variables
        orthant = 2 .* Int.(collect(Iterators.flatten(pattern))) .- 1
        A_flipped = Diagonal(orthant) * A
        b_flipped = b .* orthant

        # Add constraints: W_scaled * x + b_scaled >= epsilon (small positive)
        epsilon = 1e-6
        con_refs = @constraint(fm.model, A_flipped * fm.x .+ b_flipped .>= epsilon)

        optimize!(fm.model)
        status = termination_status(fm.model)

        # Delete constraints for next iteration
        foreach(c -> delete(fm.model, c), con_refs)
        return status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
    end

    if isnothing(epoch)
        load_path = "data/params.jlser"
    else
        load_path = @sprintf("data/params_%04d.jlser", epoch)
    end
    params = deserialize(load_path)

    L = size(layer_sizes)[1] - 1
    partition_tree = Vector{Dict{UInt128, PartitionEntry}}(undef, L)
    partition_layer = Dict{UInt128, PartitionEntry}()
    
    W_init = params["W_$(L)"][neuron_index:neuron_index, :]
    b_init = params["b_$(L)"][neuron_index:neuron_index]
    phi_init = -pinv(W_init) * b_init
    partition_layer[0] = PartitionEntry(phi_init, Vector{BitVector}(), W_init, b_init, Matrix{Float64}(undef, 0, size(W_init)[2]), Vector{Float64}(), "Boundary")
    partition_tree[1] = partition_layer

    for l in L-2:-1:0
        fm = init_fm(size(params["W_$(l+1)"])[2]-1)
        partition_layer = Dict{UInt128, PartitionEntry}()
        for (_, super_partition) in partition_tree[L-l-1]
            for layer_pattern in all_activation_patterns(layer_sizes[l+2:l+2])
                pattern =  vcat(layer_pattern, super_partition.pattern)

                D = Diagonal(layer_pattern[1])
                W = params["W_$(l+1)"]
                b = params["b_$(l+1)"]

                W_hat = super_partition.W_hat * D * W
                b_hat = super_partition.b_hat + (super_partition.W_hat * D * b)
                W_tilde = vcat(W, super_partition.W_tilde * D * W)
                b_tilde = vcat(b, super_partition.b_tilde + (super_partition.W_tilde * D * b))

                phi = -pinv(W_hat) * b_hat
                if phi != [0.0, 0.0] # If non-void and non-null, perform a projection from the space of the CAB plane and repeat LP test to test for boundary region
                    Q, _ = qr([phi I])
                    phi_ortho = Q[:, 2:end]
                    W_tilde_proj = W_tilde * phi_ortho
                    b_tilde_proj = (W_tilde * phi) + b_tilde
                    boundary_bool = LP_feasability(fm, W_tilde_proj, b_tilde_proj, pattern)
                    if boundary_bool
                        partition_layer[foldl((acc, b) -> (acc << 1) | b, Iterators.flatten(pattern); init=UInt128(0))] = PartitionEntry(phi, pattern, W_hat, b_hat, W_tilde, b_tilde, "Boundary")
                    end
                end
            end
        end
        partition_tree[L - l] = partition_layer
        empty!(fm.model)
    end

    if to_save
        if isnothing(epoch)
            save_path = "data/partition_tree.jlser"
        else
            save_path = @sprintf("data/partition_tree_%04d.jlser", epoch)
        end
        println("Saving CAB tree (boundary only) to $save_path using Serialization")
        open(save_path, "w") do io
            serialize(io, partition_tree)
        end
    end
    return partition_tree
end

"""
    get_partition_tree(path::String)

Deserializes the `.jlser` file at `path`, prints each partition's CAB position vector and tag
"""
function get_partition_tree(epoch)
    load_path = @sprintf("data/partition_tree_%04d.jlser", epoch)
    partition_tree = deserialize(load_path)
    for partition_layer in partition_tree
        for k in sort(collect(keys(partition_layer)))
            v = partition_layer[k]
            println("$k => phi: ", v.phi, ", tag: ", v.tag)
        end
    end
    return partition_tree
end


"""
    calculate_partition_all()

Calculates the partitions in the input layer only, but it does so for all neurons in the network
"""

function calculate_partition_all(layer_sizes::Vector{Int}, epoch::Union{Int, Nothing} = nothing, to_save::Bool = true)
    L = size(layer_sizes)[1] - 1
    partition_neuron_table = Vector{Vector{Dict{UInt128, PartitionEntry}}}(undef, L)
    for l in 1:L
        partition_neuron_layer = Vector{Dict{UInt128, PartitionEntry}}(undef, layer_sizes[l+1])
        for i in 1:layer_sizes[l+1]
            partition_neuron_layer[i] = calculate_partition_tree(layer_sizes[1:l+1], i, epoch, false)[end]
        end
        partition_neuron_table[l] = partition_neuron_layer
    end

    if to_save
        if isnothing(epoch)
            save_path = "data/partition_neuron_table.jlser"
        else
            save_path = @sprintf("data/partition_neuron_table_%04d.jlser", epoch)
        end
        println("Saving CAB neuron table (including non-boundary) to $save_path using Serialization")
        open(save_path, "w") do io
            serialize(io, partition_neuron_table)
        end
    end
    return partition_neuron_table
end

"""
    calculate_boundary_all()

Calculates the boundary partitions in the input layer only, but it does so for all neurons in the network
"""

function calculate_boundary_all(layer_sizes::Vector{Int}, epoch::Union{Int, Nothing} = nothing, to_save::Bool = true)
    L = size(layer_sizes)[1] - 1
    partition_neuron_table = Vector{Vector{Dict{UInt128, PartitionEntry}}}(undef, L)
    for l in 1:L
        partition_neuron_layer = Vector{Dict{UInt128, PartitionEntry}}(undef, layer_sizes[l+1])
        for i in 1:layer_sizes[l+1]
            partition_neuron_layer[i] = calculate_boundary_tree(layer_sizes[1:l+1], i, epoch, false)[end]
        end
        partition_neuron_table[l] = partition_neuron_layer
    end

    if to_save
        if isnothing(epoch)
            save_path = "data/partition_neuron_table.jlser"
        else
            save_path = @sprintf("data/partition_neuron_table_%04d.jlser", epoch)
        end
        println("Saving CAB neuron table (boundary only) to $save_path using Serialization")
        open(save_path, "w") do io
            serialize(io, partition_neuron_table)
        end
    end

    return partition_neuron_table
end

"""
    get_partition_all(path::String)

Deserializes the `.jlser` file at `path`, prints each neuron's CAB position vector and tag for each partition
"""

function get_partition_all(epoch)
    # --- Load File ---
    if isnothing(epoch)
        load_path = "data/partition_neuron_table.jlser"
    else
        load_path = @sprintf("data/partition_neuron_table_%04d.jlser", epoch)
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

function plot_CAB_frame!(ax::Axis, neuron_layer::Int, neuron_index::Int, partition_neuron_table:: Vector{Vector{Dict{UInt128, PartitionEntry}}}, epoch::Int)
    # --- Data Setup ---
    x, y = CAB_GRID.x, CAB_GRID.y
    points = CAB_GRID.points
    points_T = CAB_GRID.points_T
    grid_size = size(CAB_GRID.Xg) 

    mask = falses(size(points, 1))
    z = Vector{Float64}(undef, size(points, 1))

    # --- Clear Previous Plot ---
    empty!(ax.scene.plots)
    ax.title = isnothing(epoch) ? latexstring("z^{[$neuron_layer]}_$neuron_index \\text{ with CAB}") : latexstring("z^{[$neuron_layer]}_{$neuron_index} \\text{ with CAB at epoch $epoch}")
    println("Creating figure for epoch $epoch")

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
    plot_CAB_all(weights::Dict, layer_sizes::Vector{Int})

Plots all activation boundaries for a 2D input network.
"""

function plot_CAB_all(layer_sizes::Vector{Int}, neuron_layer::Int, neuron_index::Int, epoch::Union{Int, Nothing} = nothing, to_save::Bool = true)
    # --- Load File ---
    if isnothing(epoch)
        load_path = "data/partition_neuron_table.jlser"
    else
        load_path = @sprintf("data/partition_neuron_table_%04d.jlser", epoch)
    end
    partition_neuron_table = deserialize(load_path)
    
    # --- Figure Setup ---
    title_text = isnothing(epoch) ? latexstring("z^{[$neuron_layer]}_$neuron_index \\text{ with CAB}") : latexstring("z^{[$neuron_layer]}_{$neuron_index} \\text{ with CAB at epoch $epoch}")
    fig = Figure(size = (900, 600))
    ax = Axis(fig[1, 1],
        title = title_text,
        xlabel = L"x_1",
        ylabel = L"x_2",
        aspect = DataAspect(),
        limits = ((-5, 5), (-5, 5))
    )

    # --- Call shared plotting logic ---
    plot_CAB_frame!(ax, neuron_layer, neuron_index, partition_neuron_table, epoch)

    # --- Legend for Layers (recomputed here for fig) ---
    colors = [get(ColorSchemes.turbid, i/(length(layer_sizes)-2)) for i in 0:(length(layer_sizes)-2)]

    for (layer_idx, col) in enumerate(colors)
        label_idx = length(colors) - layer_idx + 1
        lines!(ax, [NaN], [NaN], color=col, linewidth=2, label="Layer $label_idx")
    end
    axislegend(ax, position=:lt)

    # --- Colorbars ---
    subgrid = fig[1, 2] = GridLayout()
    color_limits = (-10, 10)
    Colorbar(subgrid[1, 1], colormap = reverse(cgrad(ColorSchemes.Reds)),
             limits = color_limits, label = "Boundary", width = 20, height = Relative(0.9))
    Colorbar(subgrid[1, 2], colormap = reverse(cgrad(ColorSchemes.Purples)),
             limits = color_limits, label = "Null", width = 20, height = Relative(0.9))
    Colorbar(subgrid[1, 3], colormap = reverse(cgrad(ColorSchemes.Blues)),
             limits = color_limits, label = "Non-Boundary", width = 20, height = Relative(0.9))

    # --- Save Figure ---
    if to_save
        if isnothing(epoch)
            save_path = "CAB_plot.png"
        else
            save_path = @sprintf("plot_store/CAB_plot_%04d.png", epoch)
        end
        println("Saving CAB plot to $save_path")
        save(save_path, fig)
    end

    return fig
end

function create_animation(layer_sizes::Vector{Int}, total_epoch::Int; output_path::String = "CAB_animation.mp4", framerate::Int = 10)
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

    function make_colorbar(parent, cmap, lbl)
        Colorbar(parent, colormap = reverse(cgrad(cmap)), limits = (-10, 10), label = lbl, width = 10, height = Relative(0.9), flip_vertical_label = true)
    end

    make_colorbar(subgrid[1, 1], ColorSchemes.Reds, "Boundary")
    make_colorbar(subgrid[1, 2], ColorSchemes.Purples, "Null")
    make_colorbar(subgrid[1, 3], ColorSchemes.Blues, "Non-Boundary")

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
    record(fig, output_path, 0:total_epoch; framerate = framerate) do epoch
        partition_neuron_table = deserialize(@sprintf("data/partition_neuron_table_%04d.jlser", epoch))
        for row in 1:length(layer_sizes)-1
            neuron_layer = length(layer_sizes) - row
            for neuron_index in 1:layer_sizes[neuron_layer+1]
                plot_CAB_frame!(axes_grid[row][neuron_index], neuron_layer, neuron_index, partition_neuron_table, epoch)
            end
        end
    end

    println("Animation saved to $output_path") # Note that things are saved live in the previous block
end

"""
    plot_partition_count(total_epochs::Int, to_save = true) -> fig with 2 plots

    One plot of number of partitions of the input space at each epoch, another of the number of boundary partitions at each epoch
"""
function plot_partition_count(total_epochs::Int, neuron_layer::Union{Int, Nothing} = nothing, neuron_index::Union{Int, Nothing} = nothing, to_save = true)
    partition_counts = Int[]
    boundary_counts = Int[]

    for epoch in 0:total_epochs
        file = @sprintf("data/partition_neuron_table_%04d.jlser", epoch)
        data = deserialize(file)

        if isnothing(neuron_layer) || isnothing(neuron_index)
            partitions = data[end][1] # Topmost neuron
        else
            partitions = data[neuron_layer][neuron_index] # Chosen partition
        end

        push!(partition_counts, length(partitions)) # Count Partitions

        boundary_count = count(partition -> partition.tag == "Boundary", values(partitions)) # Count Boundary Partitions
        push!(boundary_counts, boundary_count)
    end

    fig = Figure(resolution = (1500, 500))

    ax1 = Axis(fig[1, 1], title = "Number of Partitions", xlabel = "Epoch", ylabel = "Partition Count")
    lines!(ax1, 0:total_epochs, partition_counts)

    ax2 = Axis(fig[1, 2], title = "Number of Boundary Partitions", xlabel = "Epoch", ylabel = "Boundary Partition Count")
    lines!(ax2, 0:total_epochs, boundary_counts)
    
    ax3 = Axis(fig[1, 3], title = "Proportion of Boundary Partitions", xlabel = "Epoch", ylabel = "Boundary Partition Percentage")
    lines!(ax3, 0:total_epochs, 100 .* boundary_counts ./ partition_counts)
    #Optional save
    if to_save
        save_path = "plot_store/partition_count.png"
        println("Saving Partition Count plot to $save_path")
        save(save_path, fig)
    end
    return fig
end

end # module