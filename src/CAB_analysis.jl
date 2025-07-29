module CAB_analysis

# Import Packages
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

export save_params, get_params, calculate_partition_simple, get_partition_simple, calculate_partition_tree, calculate_boundary_tree, get_partition_tree, calculate_partition_all, calculate_boundary_all, get_partition_all, plot_CAB_all, create_animation, plot_partition_count

# Used in the LP feasability test
mutable struct FeasibilityModel
    model::Model
    x::Vector{VariableRef}
    epsilon::Float64
end

#Used to store partition data
struct PartitionEntry
    phi::Vector{Float64}
    pattern::Vector{BitVector}
    W_hat::Matrix{Float64}
    b_hat::Vector{Float64}
    W_tilde::Matrix{Float64}
    b_tilde::Vector{Float64}
    tag::String
end

"""
    save_params(model::PyObject, epoch=nothing)

Given a PyTorch model (e.g. `torch.nn.Sequential`), save weight matrices and bias vectors to '.jlser' file at 'path'.
"""
function save_params(model::PyObject, epoch=nothing, to_save = true)
    torch = pyimport("torch")
    params = Dict{String, Any}()
    layer_index = 1

    # Loop through layers
    for layer in model
        if pyisinstance(layer, torch.nn.Linear)
            params["W_$layer_index"] = Array(layer.weight.detach().numpy())
            params["b_$layer_index"] = Array(layer.bias.detach().numpy())
            layer_index += 1
        end
    end

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

"""
    get_params(epoch) -> params::Dict{String, Any}

Deserializes the `.jlser` file at `path`, prints each entry's key, shape, and element type,
and returns the dictionary for use in Python.
"""
function get_params(epoch) 
    load_path = @sprintf("data/params_%04d.jlser", epoch)
    params = deserialize(load_path)

    # Sort to prevent random ordering
    for k in sort(collect(keys(params)))
        v = params[k]
        println("$k => value: ", v)
    end

    return params
end

"""
    unwrap_affine(pattern::Vector{BitVector}, epoch)

Given params and a list of ReLU activation masks, computes the affine function `f(x) = A * x + b`
for that region.
"""
function unwrap_affine(pattern::Vector{BitVector}, epoch)
    load_path = @sprintf("data/params_%04d.jlser", epoch)
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


# I am considering improving my data structuring so the below may become redundant.
"""
    all_activation_patterns(sizes::Vector{Int}) -> Iterator

Generates all possible ReLU activation patterns for each layer size.
Returns an iterator of activation patterns as `Vector{BitVector}.
"""
function all_activation_patterns(sizes::Vector{Int})
    function binary_patterns(n::Int)
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

    pattern_lists = [binary_patterns(n) for n in sizes]
    return (collect(p) for p in Iterators.product(pattern_lists...))
end

"""
    calculate_CAB_LP(layer_sizes::Vector{Int}, epoch=nothing)

Calculates the CAB position vector in each partition.
Performs a feasability test using linear programming to determine whether a partition is void.
Subsequently performs a projection and another feasability test to determine whether a partition contains a boundary or not.
Saves all results as a in a FeasibilityModel struct in the '.jlser' file at 'path'.
"""

function calculate_partition_simple(layer_sizes::Vector{Int}, epoch=nothing, to_save::Bool = true)

    function init_fm(x_size::Int; solver=HiGHS.Optimizer, epsilon=1e-6)
        model = Model(solver)
        set_silent(model)
        @variable(model, x[1:x_size])
        return FeasibilityModel(model, x, epsilon)
    end

    fm1 = init_fm(layer_sizes[1])
    fm2 = init_fm(layer_sizes[1]-1)

    function LP_feasability(fm::FeasibilityModel, A::Matrix{Float64}, b::Vector{Float64}, pattern::Vector{BitVector})
        # Input variables
        orthant = orthant = 2 .* Int.(collect(Iterators.flatten(pattern))) .- 1
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

    partitions = Dict{UInt128, PartitionEntry}()

    for pattern in all_activation_patterns(layer_sizes[2:end-1])
        W_hat, b_hat, W_tilde, b_tilde = unwrap_affine(pattern, epoch)
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
    partitions = deserialize(path)

    for k in sort(collect(keys(partitions)))
        v = partitions[k]
        println("$k => phi: ", v.phi, ", tag: ", v.tag)
    end

    return partitions
end

"""
    calculate_partition_tree(layer_sizes::Vector{Int}, neuron_index::Int, epoch=nothing)

Calculates the CAB of a Neuron in all lower layers, latent and otherwise. Ignores void partitions effectively
"""

function calculate_partition_tree(layer_sizes::Vector{Int}, neuron_index::Int, epoch = nothing, to_save::Bool = true)

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
    calculate_boundary_tree(layer_sizes::Vector{Int}, neuron_index::Int, epoch = nothing)

Calculates the CAB of a Neuron in all lower layers, latent and otherwise. Ignores void and non-boundary partitions effectively
"""

function calculate_boundary_tree(layer_sizes::Vector{Int}, neuron_index::Int, epoch = nothing, to_save::Bool=true)

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

function calculate_partition_all(layer_sizes::Vector{Int}, epoch = nothing, to_save::Bool = true)
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

function calculate_boundary_all(layer_sizes::Vector{Int}, epoch = nothing, to_save::Bool = true)
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
    load_path = @sprintf("data/partition_neuron_table_%04d.jlser", epoch)
    partition_neuron_table = deserialize(load_path)
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

function plot_CAB_frame!(ax::Axis, neuron_layer::Int, neuron_index::Int, epoch)
    if isnothing(epoch)
        load_path = "data/partition_neuron_table.jlser"
    else
        load_path = @sprintf("data/partition_neuron_table_%04d.jlser", epoch)
    end
    partition_neuron_table = deserialize(load_path)

    x = LinRange(-5, 5, 200)
    y = LinRange(-5, 5, 200)
    Xg = repeat(reshape(x, :, 1), 1, length(y))
    Yg = repeat(reshape(y, 1, :), length(x), 1)
    points = hcat(vec(Xg), vec(Yg))
    mask = falses(size(points, 1))
    z = Vector{Float64}(undef, size(points, 1))

    # Clear previous contents of the axis
    empty!(ax.scene.plots)
    ax.title = isnothing(epoch) ? latexstring("z^{[$neuron_layer]}_$neuron_index \\text{ with CAB}") : latexstring("z^{[$neuron_layer]}_{$neuron_index} \\text{ with CAB} at epoch $epoch}")
    

    color_limits = (-10, 10)
    colors = [get(ColorSchemes.turbid, i/(length(partition_neuron_table)-1)) for i in 0:(length(partition_neuron_table)-1)]

    Zg              = fill(NaN, size(Xg))
    Zg_boundary     = fill(NaN, size(Xg))
    Zg_null         = fill(NaN, size(Xg))
    Zg_non_boundary = fill(NaN, size(Xg))

    println("Creating figure for epoch $epoch")

    for (_, partition) in partition_neuron_table[neuron_layer][neuron_index]
        fill!(mask, false)
        if isempty(partition.pattern) || isempty(partition.W_tilde) || isempty(partition.b_tilde)
            mask .= true
        else
            orthant = 2 .* Int.(reduce(vcat, partition.pattern)) .- 1
            W_tilde_flipped = Diagonal(orthant) * partition.W_tilde
            b_tilde_flipped = partition.b_tilde .* orthant
            mask .= vec(all(W_tilde_flipped * points' .+ b_tilde_flipped .> 0, dims=1))
        end
        mul!(z, points, vec(partition.W_hat))
        z .+= partition.b_hat

        if partition.tag == "Boundary"
            Zg_boundary[mask] .= z[mask]
            mul!(z, points, vec(partition.W_hat))
            z .+= partition.b_hat
            Zg[mask] .= z[mask]
        elseif partition.tag == "Null"
            Zg_null[mask] .= z[mask]
        else
            Zg_non_boundary[mask] .= z[mask]
        end
    end

    heatmap!(ax, x, y, Zg_boundary; colormap = reverse(cgrad(ColorSchemes.Reds)), colorrange = color_limits, alpha=0.8)
    heatmap!(ax, x, y, Zg_null;     colormap = reverse(cgrad(ColorSchemes.Purples)), colorrange = color_limits, alpha=0.8)
    heatmap!(ax, x, y, Zg_non_boundary; colormap = reverse(cgrad(ColorSchemes.Blues)), colorrange = color_limits, alpha=0.8)
    contour!(ax, x, y, Zg; levels=[0], linewidth=2, color=colors[neuron_layer])

    for (layer_idx, partition_neuron_layer) in enumerate(partition_neuron_table[1:neuron_layer-1])
        for partition_neuron in partition_neuron_layer
            Zg .= NaN
            for (_, partition) in partition_neuron
                if partition.tag == "Boundary"
                    if isempty(partition.pattern)
                        mask .= true
                    else
                        orthant = 2 .* Int.(reduce(vcat, partition.pattern)) .- 1
                        W_tilde_flipped = Diagonal(orthant) * partition.W_tilde
                        b_tilde_flipped = partition.b_tilde .* orthant
                        mask .= vec(all(W_tilde_flipped * points' .+ b_tilde_flipped .> 0, dims=1))
                    end
                    mul!(z, points, vec(partition.W_hat))
                    z .+= partition.b_hat
                    Zg[mask] .= z[mask]
                end
            end
            contour!(ax, x, y, Zg; levels=[0], linewidth=2, color=colors[layer_idx])
        end
    end
end

"""
    plot_CAB_all(weights::Dict, layer_sizes::Vector{Int})

Plots all activation boundaries for a 2D input network.
"""

function plot_CAB_all(layer_sizes::Vector{Int}, neuron_layer::Int, neuron_index::Int, epoch = nothing, to_save::Bool = true)
    # --- Figure Setup ---
    title_text = isnothing(epoch) ? L"z^{[$neuron_layer]}_$neuron_index with CAB" : L"Pre-activation z^{[$neuron_layer]}_$neuron_index with CAB at (epoch = $epoch)"
    fig = Figure(size = (900, 600))
    ax = Axis(fig[1, 1],
        title = title_text,
        xlabel = L"x_1",
        ylabel = L"x_2",
        aspect = DataAspect(),
        limits = ((-5, 5), (-5, 5))
    )

    # --- Call shared plotting logic ---
    plot_CAB_frame!(ax, neuron_layer, neuron_index, epoch)

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

function create_animation(layer_sizes::Vector{Int}, total_epoch::Int;
                          output_path::String = "CAB_animation.mp4",
                          framerate::Int = 10)
    # === Figure ===
    fig = Figure(size = (1400, 1000), figure_padding=5)
    Label(fig[0, 2], "Pre-activations and CAB's of all neurons", fontsize = 28, tellwidth = false, tellheight = true)

    # Reduce gaps between rows and columns
    rowgap!(fig.layout, 5)         # 5px vertical spacing
    colgap!(fig.layout, 5)         # 5px horizontal spacing

    # === Store Axes in Grid=== 
    axes_grid = Vector{Vector{Axis}}(undef, length(layer_sizes)-1)

    # --- Legend for Layers (recomputed here for fig) ---
    colors = [get(ColorSchemes.turbid, i/(length(layer_sizes)-2)) for i in 0:(length(layer_sizes)-2)]
    dummy_axis = Axis(fig.scene)  # Create axis not added to layout
    legend_lines = [lines!(dummy_axis, [NaN], [NaN], color=col, linewidth=2) for col in colors]
    Legend(fig[1, 1], legend_lines, ["Layer $i" for i in 1:length(colors)]; title = "Layers")
    colsize!(fig.layout, 1, 100)

    # === Top Layer=== #
    axes_grid[1] = Vector{Axis}(undef, layer_sizes[end])
    top_grid = fig[1, 2] = GridLayout()
    colgap!(top_grid, 5)
    rowsize!(fig.layout, 1, Relative(0.4))

    for neuron_index in 1:layer_sizes[end]
            axes_grid[1][neuron_index] = Axis(top_grid[1, neuron_index], aspect = DataAspect())
            colsize!(top_grid, neuron_index, Auto())
        end
    # === Colorbars=== 
    subgrid = fig[1, 3] = GridLayout()
    colsize!(fig.layout, 3, 100)
    Colorbar(subgrid[1, 1], colormap = reverse(cgrad(ColorSchemes.Reds)),
             limits = (-10, 10), label = "Boundary", width = 20, height = Relative(0.9))
    Colorbar(subgrid[1, 2], colormap = reverse(cgrad(ColorSchemes.Purples)),
             limits = (-10, 10), label = "Null", width = 20, height = Relative(0.9))
    Colorbar(subgrid[1, 3], colormap = reverse(cgrad(ColorSchemes.Blues)),
             limits = (-10, 10), label = "Non-Boundary", width = 20, height = Relative(0.9))

    for row in 2:length(layer_sizes)-1
        neuron_layer = length(layer_sizes) - row

        row_grid = fig[row, 2] = GridLayout()
        colgap!(row_grid, 5)
        rowsize!(fig.layout, row, Auto())
        axes_grid[row] = Vector{Axis}(undef, layer_sizes[neuron_layer + 1])

        for neuron_index in 1:layer_sizes[neuron_layer+1]
            axes_grid[row][neuron_index] = Axis(row_grid[1, neuron_index], aspect = DataAspect())
            # Make columns uniform width
            colsize!(row_grid, neuron_index, Auto())
        end
    end

    # === Animation Recording ===
    record(fig, output_path, 0:(total_epoch-1); framerate = framerate) do epoch
        # Update all plots
        for row in 1:length(layer_sizes)-1
            neuron_layer = length(layer_sizes) - row
            for neuron_index in 1:layer_sizes[neuron_layer+1]
                plot_CAB_frame!(axes_grid[row][neuron_index], neuron_layer, neuron_index, epoch)
            end
        end
    end

    println("Animation saved to $output_path")
end

function create_CAB_dashboard(layer_sizes::Vector{Int}, total_epoch::Int)
    fig = Figure(resolution = (1000, 700))
    ax = Axis(fig[1, 1],
              title = "Analytical CAB",
              xlabel = L"x_1", ylabel = L"x_2",
              aspect = DataAspect(), limits = ((-5, 5), (-5, 5)))

    # Colorbars
    subgrid = fig[1, 2] = GridLayout()
    Colorbar(subgrid[1, 1], colormap = reverse(cgrad(ColorSchemes.Reds)),
             limits = (-10, 10), label = "Boundary", width = 20, height = Relative(0.9))
    Colorbar(subgrid[1, 2], colormap = reverse(cgrad(ColorSchemes.Purples)),
             limits = (-10, 10), label = "Null", width = 20, height = Relative(0.9))
    Colorbar(subgrid[1, 3], colormap = reverse(cgrad(ColorSchemes.Blues)),
             limits = (-10, 10), label = "Non-Boundary", width = 20, height = Relative(0.9))

    # UI elements
    slider = Slider(fig[2, 1], range = 0:total_epoch-1, startvalue=0)
    play_button = Button(fig[2, 2], label="▶ Play")

    current_epoch = Observable(0)
    playing = Observable(false)

    # When slider moves, update current_epoch
    on(slider.value) do val
        current_epoch[] = Int(val)
        plot_CAB_frame!(ax, 0, 1, current_epoch[])
    end

    # Button toggles play/pause
    on(play_button.clicks) do _
        playing[] = !playing[]
        play_button.label[] = playing[] ? "⏸ Pause" : "▶ Play"
    end

    # Animation loop (as a task)
    @async begin
        while isopen(fig)
            if playing[]
                next_epoch = (current_epoch[] + 1) % total_epoch
                current_epoch[] = next_epoch
                slider.value[] = next_epoch
                plot_CAB_frame!(ax, 0, 1, current_epoch[])
            end
            sleep(0.2)  # control animation speed
        end
    end

    plot_CAB_frame!(ax, 0, 1, neuron_index, current_epoch[]) # Initial plot
    display(fig)
    return fig
end

function plot_partition_count(total_epochs)
    first_elem_counts = Int[]
    boundary_counts = Int[]

    for epoch in 0:total_epochs
        file = @sprintf("data/partition_neuron_table_%04d.jlser", epoch)
        if isfile(file)
            data = deserialize(file)

            # The vector that is the first element
            first_elem = data[1]
            push!(first_elem_counts, length(first_elem))

            # Count how many elements have .Tag == "Boundary"
            boundary_count = count(x -> getfield(x, :Tag, nothing) == "Boundary", first_elem)
            push!(boundary_counts, boundary_count)
        else
            @warn "File not found: $file"
            push!(first_elem_counts, NaN)
            push!(boundary_counts, NaN)
        end
    end

    fig = Figure(resolution = (1000, 500))

    ax1 = Axis(fig[1, 1], title = "Size of First Element", xlabel = "Epoch", ylabel = "Count")
    lines!(ax1, 0:total_epochs, first_elem_counts)

    ax2 = Axis(fig[1, 2], title = "Boundary Tags", xlabel = "Epoch", ylabel = "Boundary Count")
    lines!(ax2, 0:total_epochs, boundary_counts)

    return fig
end

end # module