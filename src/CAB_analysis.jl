module CAB_analysis

# Import Packages
using PyCall
using Serialization
using LinearAlgebra
using JuMP
using HiGHS
using Plots

export save_params, get_params, calculate_partitions, get_partitions, plot_CAB

# Used in the LP feasability test
mutable struct FeasibilityModel
    model::Model
    x::Vector{VariableRef}
    epsilon::Float64
end

#Used to store partition data
struct PartitionEntry{N}
    phi::Vector{Float64}
    pattern::NTuple{N, BitVector}
    W_tilde::Matrix{Float64}
    b_tilde::Vector{Float64}
    W_hidden_stack::Matrix{Float64}
    b_hidden_stack::Vector{Float64}
    tag::String
end

"""
    save_params(model::PyObject, path::String)

Given a PyTorch model (e.g. `torch.nn.Sequential`), save weight matrices and bias vectors to '.jlser' file at 'path'.
"""
function save_params(model::PyObject, path::String)
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

    println("Saving params to $path using Serialization")
    open(path, "w") do io
        serialize(io, params)
    end
end

"""
    get_params(path::String) -> params::Dict{String, Any}

Deserializes the `.jlser` file at `path`, prints each entry's key, shape, and element type,
and returns the dictionary for use in Python.
"""
function get_params(path::String)
    params = deserialize(path)

    # Sort to prevent random ordering
    for k in sort(collect(keys(params)))
        v = params[k]
        println("$k => value: ", v)
    end

    return params
end

"""
    unwrap_affine(params::Dict, pattern::Vector{BitVector}) -> (W_tilde::Matrix, b_tilde::Matrix, W_hidden_stack::Matrix, b_hidden_stack::Matrix)

Given params and a list of ReLU activation masks, computes the affine function `f(x) = A * x + b`
for that region.
"""
function unwrap_affine(path::String, pattern::NTuple{N, BitVector}) where {N}
    params = deserialize(path)
    L = N + 1  # Total layers

    # Initialize W_tilde and b_tilde
    W_tilde = params["W_1"]
    b_tilde = params["b_1"]

    # Sequences (store all intermediate W_tilde and b_tilde)
    W_hidden_seq = [W_tilde]
    b_hidden_seq = [b_tilde]

    # Loop through layers
    for l in 2:L
        D = Diagonal(Float64.(pattern[l-1]))
        W = params["W_$(l)"]
        b = params["b_$(l)"]

        W_tilde = W * D * W_tilde
        b_tilde = W * D * b_tilde .+ b

        # Append current W_tilde and b_tilde to the sequences
        push!(W_hidden_seq, W_tilde)
        push!(b_hidden_seq, b_tilde)
    end

    # This matrix and vector output the result in all hidden layers of the matrix, but not the actual output.
    W_hidden_stack = reduce(vcat, W_hidden_seq[1:end-1])
    b_hidden_stack = reduce(vcat, b_hidden_seq[1:end-1])

    return W_tilde, b_tilde, W_hidden_stack, b_hidden_stack
end


# I am considering improving my data structuring so the below may become redundant.
"""
    all_activation_patterns(sizes::Vector{Int}) -> Iterator

Generates all possible ReLU activation patterns for each layer size.
Returns an iterator of activation patterns as `Vector{BitVector}.
"""
function all_activation_patterns(sizes::Vector{Int})
    function binary_patterns(n)
        return [BitVector(digits(i, base=2, pad=n)) for i in 0:(2^n - 1)]
    end

    pattern_lists = [binary_patterns(n) for n in sizes]
    return Iterators.product(pattern_lists...)
end

"""
    calculate_CAB_LP(param_path::String, partition_path::String, input_dim::Int, layer_sizes::Vector{Int})

Calculates the CAB position vector in each partition.
Performs a feasability test using linear programming to determine whether a partition is void.
Subsequently performs a projection and another feasability test to determine whether a partition contains a boundary or not.
Saves all results as a in a FeasibilityModel struct in the '.jlser' file at 'path'.
"""

function calculate_partitions(param_path::String, partition_path::String, input_dim::Int, layer_sizes::Vector{Int})

    function init_fm(x_size::Int; solver=HiGHS.Optimizer, epsilon=1e-6)
        model = Model(solver)
        set_silent(model)
        @variable(model, x[1:x_size])
        return FeasibilityModel(model, x, epsilon)
    end

    fm1 = init_fm(input_dim)
    fm2 = init_fm(input_dim-1)

    function LP_feasability(fm::FeasibilityModel, A::Matrix{Float64}, b::Vector{Float64}, pattern::NTuple{N, BitVector}) where {N}
        # Input variables
        orthant = 2 .* 2 .* Int.(reduce(vcat, pattern)) .- 1
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

    for pattern in all_activation_patterns(layer_sizes)
        W_tilde, b_tilde, W_hidden_stack, b_hidden_stack = unwrap_affine(param_path, pattern)
        phi = -pinv(W_tilde) * b_tilde
        nonvoid_bool = LP_feasability(fm1, W_hidden_stack, b_hidden_stack, pattern)
        tag = "N/A"
        if nonvoid_bool
            if phi == [0.0, 0.0]
                tag = "Null"
            else    # If non-void and non-null, perform a projection from the space of the CAB plane and repeat LP test to test for boundary region
                Q, _ = qr([phi I])
                phi_ortho = Q[:, 2:end]
                W_hidden_stack_proj = W_hidden_stack * phi_ortho
                b_hidden_stack_proj = (W_hidden_stack * phi) + b_hidden_stack
                boundary_bool = LP_feasability(fm2, W_hidden_stack_proj, b_hidden_stack_proj, pattern)
                if boundary_bool
                    tag = "Boundary"
                else
                    tag = "Non-Boundary"
                end
            end
        else
            if phi == [0.0, 0.0]
                tag = "Nullvoid" # If phi is also null
            else
                tag = "Void"
            end
        end
        partitions[foldl((acc, b) -> (acc << 1) | b, Iterators.flatten(pattern); init=UInt128(0))] = PartitionEntry(phi, pattern, W_tilde, b_tilde, W_hidden_stack, b_hidden_stack, tag)
    end

    println("Saving CAB to $partition_path using Serialization")
    open(partition_path, "w") do io
        serialize(io, partitions)
    end

end


"""
    get_partitions(path::String)

Deserializes the `.jlser` file at `path`, prints each partition's CAB position vector and tag
"""
function get_partitions(path::String)
    partitions = deserialize(path)

    for k in sort(collect(keys(partitions)))
        v = partitions[k]
        println("$k => phi: ", v.phi, ", tag: ", v.tag)
    end

    return partitions
end

function calculate_CAB_tree(param_path::String, partition_tree_path::String, input_dim::Int, layer_sizes::Vector{Int})

end

"""
    plot_CAB(weights::Dict, layer_sizes::Vector{Int})

Plots activation boundaries for a 2D input network.
"""

function plot_CAB(path::String, layer_sizes::Vector{Int})
    partitions = deserialize(path)
    x = LinRange(-5, 5, 500)
    y = LinRange(-5, 5, 500)
    Xg = repeat(reshape(x, :, 1), 1, length(y))
    Yg = repeat(reshape(y, 1, :), length(x), 1)
    points = hcat(vec(Xg), vec(Yg))  # (N × 2) matrix, where N = 500 * 500

    # Prepare the plot
    plt = plot(; title="Analytical CAB", xlabel="x", ylabel="y", 
               xlims=(-5, 5), ylims=(-5, 5), aspect_ratio=1)

    # Preallocate for efficiency
    Zg = Array{Float64}(undef, size(Xg))
    mask = falses(size(points, 1))
    z = similar(points, size(points, 1))

    # Color map for partition types
    tag_colors = Dict(
        "Boundary" => :red,
        "Non-Boundary" => :blue,
        "Null" => :purple,
        "Void" => :black,
        "Nullvoid" => :brown
    )

    # Loop over partitions
    for (_, partition) in partitions
        color = get(tag_colors, partition.tag, :gray)  # Default to gray if unknown

        # Compute mask: points where all hidden constraints are positive
        orthant = 2 .* Int.(reduce(vcat, partition.pattern)) .- 1
        W_hidden_stack_flipped = Diagonal(orthant) * partition.W_hidden_stack
        b_hidden_stack_flipped = partition.b_hidden_stack .* orthant
        mask .= vec(all(W_hidden_stack_flipped * points' .+ b_hidden_stack_flipped .> 0, dims=1))

        # Shade region based on mask
        Zg .= NaN
        Zg[mask] .= 1  # Just fill with 1 where mask is true
        contourf!(plt, x, y, Zg, levels=[0.5, 1.5], color=color, alpha=0.3, colorbar=false)

        # If Boundary, also plot contour line where z=0
        if partition.tag == "Boundary"
            mul!(z, points, partition.phi)
            z .-= dot(partition.phi, partition.phi)
            Zg .= NaN
            Zg[mask] .= z[mask]
            contour!(plt, x, y, Zg, levels=[0], linewidth=2, color=:red, colorbar=false)
        end
    end

    savefig(plt, "CAB_plot.png")
    return plt
end

end # module