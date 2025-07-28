module CAB_analysis

# Import Packages
using PyCall
using Serialization
using LinearAlgebra
using JuMP
using HiGHS
using Plots
using Colors
using ColorSchemes
using LaTeXStrings
using Printf
using FFMPEG

export save_params, get_params, calculate_partition_simple, get_partition_simple, calculate_partition_tree, calculate_boundary_tree, get_partition_tree, calculate_boundary_all, get_partition_all, calculate_mixed_all, plot_CAB_all, create_animation

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
function save_params(model::PyObject, epoch=nothing)
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

    if !isnothing(epoch)
        save_path = @sprintf("data/params_%04d.jlser", epoch)
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

function calculate_partition_simple(layer_sizes::Vector{Int}, epoch=nothing)

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

    if !isnothing(epoch)
        save_path = @sprintf("data/partitions_%04d.jlser", epoch)
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

function calculate_partition_tree(layer_sizes::Vector{Int}, neuron_index::Int, epoch=nothing)

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
    if !isnothing(epoch)
        save_path = @sprintf("data/partition_tree_%04d.jlser", epoch)
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

function calculate_boundary_tree(layer_sizes::Vector{Int}, neuron_index::Int, epoch = nothing)

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
    if !isnothing(epoch)
        save_path = @sprintf("data/partition_tree_%04d.jlser", epoch)
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

function calculate_partition_all(layer_sizes::Vector{Int}, epoch = nothing)
    L = size(layer_sizes)[1] - 1
    partition_neuron_table = Vector{Vector{Dict{UInt128, PartitionEntry}}}(undef, L)
    for l in 1:L
        partition_neuron_layer = Vector{Dict{UInt128, PartitionEntry}}(undef, layer_sizes[l+1])
        for i in 1:layer_sizes[l+1]
            partition_neuron_layer[i] = calculate_partition_tree(layer_sizes[1:l+1], i, epoch)[end]
        end
        partition_neuron_table[l] = partition_neuron_layer
    end

    if !isnothing(epoch)
        save_path = @sprintf("data/partition_neuron_table_%04d.jlser", epoch)
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

function calculate_boundary_all(layer_sizes::Vector{Int}, epoch = nothing)
    L = size(layer_sizes)[1] - 1
    partition_neuron_table = Vector{Vector{Dict{UInt128, PartitionEntry}}}(undef, L)
    for l in 1:L
        partition_neuron_layer = Vector{Dict{UInt128, PartitionEntry}}(undef, layer_sizes[l+1])
        for i in 1:layer_sizes[l+1]
            partition_neuron_layer[i] = calculate_boundary_tree(layer_sizes[1:l+1], i, epoch)[end]
        end
        partition_neuron_table[l] = partition_neuron_layer
    end

    if !isnothing(epoch)
        save_path = @sprintf("data/partition_neuron_table_%04d.jlser", epoch)
        println("Saving CAB neuron table (boundary only) to $save_path using Serialization")
        open(save_path, "w") do io
            serialize(io, partition_neuron_table)
        end
    end

    return partition_neuron_table
end

"""
    calculate_mixed_all()

Calculates the boundary partitions in the input layer only, but it does so for all neurons in the network
"""

function calculate_mixed_all(layer_sizes::Vector{Int}, epoch = nothing)
    L = size(layer_sizes)[1] - 1
    partition_neuron_table = Vector{Vector{Dict{UInt128, PartitionEntry}}}(undef, L)
    for l in 1:L-1
        partition_neuron_layer = Vector{Dict{UInt128, PartitionEntry}}(undef, layer_sizes[l+1])
        for i in 1:layer_sizes[l+1]
            partition_neuron_layer[i] = calculate_boundary_tree(layer_sizes[1:l+1], i, epoch)[end]
        end
        partition_neuron_table[l] = partition_neuron_layer
    end

    partition_neuron_layer = Vector{Dict{UInt128, PartitionEntry}}(undef, layer_sizes[L+1])
    for i in 1:layer_sizes[L+1]
        partition_neuron_layer[i] = calculate_partition_tree(layer_sizes[1:L+1], i, epoch)[end]
    end
    partition_neuron_table[L] = partition_neuron_layer

    if !isnothing(epoch)
        save_path = @sprintf("data/partition_neuron_table_%04d.jlser", epoch)
        println("Saving CAB neuron table (boundary only unless top neuron) to $save_path using Serialization")
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

"""
    plot_CAB_all(weights::Dict, layer_sizes::Vector{Int})

Plots all activation boundaries for a 2D input network.
"""

function plot_CAB_all(epoch = nothing)
    load_path = @sprintf("data/partition_neuron_table_%04d.jlser", epoch)
    partition_neuron_table = deserialize(load_path)

    x = LinRange(-5, 5, 100)
    y = LinRange(-5, 5, 100)
    Xg = repeat(reshape(x, :, 1), 1, length(y))
    Yg = repeat(reshape(y, 1, :), length(x), 1)
    points = hcat(vec(Xg), vec(Yg))  # (N × 2) matrix
    mask = falses(size(points, 1))
    z = Vector{Float64}(undef, size(points, 1))  # preallocate for one scalar per point

    # --- Main Plot ---
    plt = plot(; 
        title = isnothing(epoch) ? "Analytical CAB" : "Analytical CAB (epoch = $epoch)",
        xlabel = L"x_1",
        ylabel = L"x_2",
        xlims = (-5, 5), 
        ylims = (-5, 5), 
        aspect_ratio = 1,
        legend=:topleft
    )  

    color_limits = (-10, 10)
    Zg_boundary = fill(NaN, size(Xg))
    Zg_null = fill(NaN, size(Xg))
    Zg_non_boundary = fill(NaN, size(Xg))

    # --- Heatmap for partitions ---
    for (_, partition) in partition_neuron_table[end][1]
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
        elseif partition.tag == "Null"
            Zg_null[mask] .= z[mask]
        else 
            Zg_non_boundary[mask] .= z[mask]
        end
    end

    heatmap!(plt, x, y, Zg_boundary, color=cgrad(ColorSchemes.Reds, rev=true), clim=color_limits, alpha=0.8, interpolate=true, colorbar=false, label=false)
    heatmap!(plt, x, y, Zg_null,     color=cgrad(ColorSchemes.Purples, rev=true), clim=color_limits, alpha=0.8, interpolate=true, colorbar=false, label=false)
    heatmap!(plt, x, y, Zg_non_boundary, color=cgrad(ColorSchemes.Blues, rev=true), clim=color_limits, alpha=0.8, interpolate=true, colorbar=false, label=false)


    # --- Contour Lines ---
    colors = [get(ColorSchemes.turbid, 1 - i/(length(partition_neuron_table)-1)) for i in 0:(length(partition_neuron_table)-1)]
    Zg = fill(NaN, size(Xg))

    for (layer_idx, partition_neuron_layer) in enumerate(Iterators.reverse(partition_neuron_table))
        for partitions in partition_neuron_layer
            Zg .= NaN
            for (_, partition) in partitions
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
            contour!(plt, x, y, Zg, levels=[0],
                linewidth=2,
                color=colors[layer_idx],
                colorbar=false,
                label="Layer $layer_idx"
            )
        end
    end
    for (layer_idx, col) in enumerate(colors)
        label_idx = length(colors) - layer_idx + 1
        plot!(plt, [NaN], [NaN], color=col, linewidth=2, label="Layer $label_idx")
    end

    # --- Colorbars (Reds, Purples, Blues) ---
    n_colors = 256
    color_values = LinRange(color_limits[1], color_limits[2], n_colors)

    function single_colorbar(cscheme)
        return heatmap(
            [1], color_values, reshape(1:length(color_values), :, 1),
            color = cgrad(cscheme, rev=true),   # <-- reversed
            xaxis = false, ylims = color_limits,
            yticks = range(color_limits[1], color_limits[2], length=5),
            legend = false, framestyle = :box,
            size = (10, 100)
        )
    end

    colorbar_red    = single_colorbar(ColorSchemes.Reds)
    colorbar_purple = single_colorbar(ColorSchemes.Purples)
    colorbar_blue   = single_colorbar(ColorSchemes.Blues)

    # Combine colorbars horizontally
    combined_cbar = plot(colorbar_red, colorbar_purple, colorbar_blue, layout = (1, 3), size=(150, 600))

    # --- Combine main plot and colorbars ---
    final_layout = @layout [a{0.8w} b{0.2w}]
    final_plot = plot(plt, combined_cbar, layout = final_layout, size = (900, 600))

    # --- Save Figure ---
    if isnothing(epoch)
        save_path = "CAB_plot.png"
    else
        save_path = @sprintf("plot_store/CAB_plot_%04d.png", epoch)
    end
    println("Saving CAB plot to $save_path")
    savefig(final_plot, save_path)
    return final_plot
end

function create_animation()
    ffmpeg_exe = FFMPEG.ffmpeg()
    run(`$ffmpeg_exe -framerate 10 -i plot_store/CAB_plot_%04d.png -c:v libx264 -pix_fmt yuv420p CAB_animation.mp4`)
end

end # module