##############################
# CAB_visualiser_empirical.jl
##############################
module CAB_visualiser_empirical

using Flux
using GLMakie
using ColorSchemes
using LaTeXStrings
using Printf
using GeometryBasics
using Meshing
using Statistics

export create_empirical_animation, plot_empirical_CAB, plot_empirical_CAB_3D

struct EmpiricalGrid
    x::LinRange{Float64}
    y::LinRange{Float64}
    Xg::Matrix{Float64}
    Yg::Matrix{Float64}
    points::Matrix{Float64}
    points_T::Matrix{Float64}
end

const EMPIRICAL_GRID = let
    res = 500
    x = LinRange(-5, 5, 500)
    y = LinRange(-5, 5, 500)
    Xg = repeat(reshape(x, :, 1), 1, length(y))
    Yg = repeat(reshape(y, 1, :), length(x), 1)
    points = hcat(vec(Xg), vec(Yg))
    points_T = points'
    EmpiricalGrid(x, y, Xg, Yg, points, points_T)
end

# Get the output size of each layer (only Dense / LayerNorm / BatchNorm)
function get_empirical_n(model::Flux.Chain, input_dim::Int)
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

# ─────────────────────────────
# Get pre-activations for each layer
# ─────────────────────────────
function get_pre_activations(model::Flux.Chain, a::Matrix{Float64})
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
        elseif layer isa WeightNorm && layer.layer isa Dense
            eff_weight = layer.g .* (layer.v ./ norm(layer.v))
            z = eff_weight * a .+ layer.layer.bias
            a = layer.layer.σ.(z)
        end
        push!(pre_activations, Dict(:z => z, :a => a))
    end
    return pre_activations
end

# ─────────────────────────────
# Plot a single neuron’s heatmap + contours
# ─────────────────────────────
function plot_empirical_frame!(ax::Axis, pre_activations::Vector{Dict}, neuron_layer::Int, neuron_index::Int, frame::Int)
    x, y = EMPIRICAL_GRID.x, EMPIRICAL_GRID.y

    grid_size = (length(y), length(x))
    
    # Pre-activation of selected neuron
    z = get(pre_activations[neuron_layer], :z, pre_activations[neuron_layer][:a])[neuron_index, :]
    Zg = reshape(z, grid_size)

    # Clear old plots
    empty!(ax.scene.plots)
    ax.title = isnothing(frame) ?
        latexstring("z^{[$neuron_layer]}_$neuron_index \\text{ empirical}") :
        latexstring("z^{[$neuron_layer]}_{$neuron_index} \\text{ empirical at frame $frame}")
    ax.xlabel = latexstring("a^{[0]}_1")
    ax.ylabel = latexstring("a^{[0]}_2")
    
    #println("Creating Figure for Neuron Layer: $neuron_layer, Neuron Index: $neuron_index, Frame: $frame")

    # Heatmap
    max_Zg = maximum(abs.(Zg))
    heatmap!(ax, x, y, Zg; colormap = reverse(cgrad(ColorSchemes.broc)), colorrange = (-max_Zg, max_Zg), alpha = 1)

    # 0-contours for this + all lower neurons
    L = length(pre_activations)
    colors = [get(ColorSchemes.turbid, (i-1)/(L-1)) for i in 1:L]
    for i in 1:neuron_layer
        for j in 1:size(pre_activations[i][:z], 1)
            if i != neuron_layer || j == neuron_index
                z_lower = pre_activations[i][:z][j, :]
                Zg_lower = reshape(z_lower, grid_size)
                #contour!(ax, x, y, Zg_lower; levels=[0.0], linewidth=2, color=colors[i])
            end
        end
    end
end

function plot_empirical_CAB(models::Vector{Flux.Chain}, neuron_layer::Int, neuron_index::Int, frame::Union{Int, Nothing} = nothing, to_save::Bool = true)
    model = models[frame+1]
    # --- Figure Setup ---
    title_text = isnothing(frame) ? latexstring("z^{[$neuron_layer]}_$neuron_index \\text{ with CAB}") : latexstring("z^{[$neuron_layer]}_{$neuron_index} \\text{ with CAB at frame $frame}")
    fig = Figure(size = (900, 600))
    ax = Axis(fig[1, 1], title = title_text, xlabel = L"a^{[0]}_1", ylabel = L"a^{[0]}_2", aspect = DataAspect(), limits = ((-5, 5), (-5, 5)))

    # --- Call shared plotting logic ---
    points_T = EMPIRICAL_GRID.points_T
    pre_activations = get_pre_activations(model, points_T)
    plot_empirical_frame!(ax, pre_activations, neuron_layer, neuron_index, frame)

    # --- Legend for Layers (recomputed here for fig) ---
    L = length(pre_activations)
    colors = [get(ColorSchemes.turbid, (i-1)/(L-1)) for i in 1:L]
    for (layer_idx, col) in enumerate(colors)
        label_idx = L + 1 - layer_idx
        lines!(ax, [NaN], [NaN], color=col, linewidth=2, label="Layer $label_idx")
    end
    axislegend(ax, position=:lt)

    # --- Colorbar ---
    Colorbar(fig[1, 2], colormap = reverse(cgrad(ColorSchemes.broc)), limits = (-10, 10), label = "Non-Boundary", width = 20, height = Relative(0.9))

    # --- Save Figure ---
    if to_save
        if isnothing(frame)
            save_path = "CAB_empirical_plot.png"
        else
            save_path = @sprintf("plot_store/CAB_empirical_plot_%04d.png", frame)
        end
        println("Saving CAB plot to $save_path")
        save(save_path, fig)
    end

    return fig
end

# ─────────────────────────────
# Animation driver
# ─────────────────────────────
function create_empirical_animation(models::Vector{Flux.Chain}, total_frame::Int; output_path::String="CAB_animation_empirical.mp4", framerate::Int=10)
    # === Dimensions ===
    n = get_empirical_n(models[1], 2)
    L = length(n)-1

    # === Figure ===
    fig = Figure(size = (1280, 720))
    rowgap!(fig.layout, 5)         # 5px vertical spacing
    colgap!(fig.layout, 5)         # 5px horizontal spacing
    
    # === Title ===
    Label(fig[0, 2], "Pre-activations and CAB's of all neurons", fontsize = 28, tellwidth = false, tellheight = true)
    rowsize!(fig.layout, 0, Auto(false))

    # === Legend for Layers ===
    colors = [get(ColorSchemes.turbid, (i-1)/(L-1)) for i in 1:L]
    dummy_axis = Axis(fig.scene)  # Create axis not added to layout
    legend_lines = [lines!(dummy_axis, [NaN], [NaN], color=col, linewidth=2) for col in colors]
    Legend(fig[1, 1], legend_lines, ["Layer $i" for i in 1:L]; title = "CAB Legend")
    colsize!(fig.layout, 1, Auto(false))

    # === Colorbar === 
    Colorbar(fig[1, 3], colormap = reverse(cgrad(ColorSchemes.broc)), limits = (-10, 10), label = "Non-Boundary", width = 20, height = Relative(0.9))
    colsize!(fig.layout, 3, Auto(false))

    # === Plots ===
    axes_grid = Vector{Vector{Axis}}(undef, L)
    for row in 1:L
        neuron_layer = L + 1 - row
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
    record(fig, output_path, 0:total_frame; framerate=framerate) do frame
        points_T = EMPIRICAL_GRID.points_T
        pre_activations = get_pre_activations(models[frame+1], points_T)
        for row in 1:L
            neuron_layer = L + 1 - row
            for neuron_index in 1:n[neuron_layer+1]
                ax = axes_grid[row][neuron_index]
                empty!(ax)
                plot_empirical_frame!(ax, pre_activations, neuron_layer, neuron_index, frame)
            end
        end
    end

    println("Empirical animation saved to $output_path")
end

function plot_empirical_CAB_3D(model::Chain, neuron_layer::Int, neuron_index::Int, frame::Union{Int, Nothing}=nothing; N::Int=100)
    # --- Grid setup ---
    x = LinRange(-50, 50, N)
    y = LinRange(-50, 50, N)
    z = LinRange(-50, 50, N)
    
    XYZ = collect(Iterators.product(x, y, z))
    XYZ = reduce(vcat, [[xi yi zi] for (xi, yi, zi) in XYZ])
    XYZ = permutedims(XYZ)   # now size is (3, N^3)
    print(typeof(XYZ))
    print(size(XYZ))
        # --- Evaluate neuron ---
    pre_activations = get_pre_activations(model, XYZ)
    
    vals = get(pre_activations[neuron_layer], :z, pre_activations[neuron_layer][:a])[neuron_index, :]
    V = reshape(vals, N, N, N)
    
    # --- Figure ---
    title_text = isnothing(frame) ? latexstring("z^{[$neuron_layer]}_$neuron_index \\text{ isosurface}") : latexstring("z^{[$neuron_layer]}_{$neuron_index} \\text{ isosurface at frame $frame}")
    
    fig = Figure(size=(900, 700))
    ax = Axis3(fig[1, 1]; xlabel=latexstring("a^{[0]}_1"),  ylabel=latexstring("a^{[0]}_2"), zlabel=latexstring("a^{[0]}_3"), title=title_text)
    
    # --- Isosurface ---
    volume!(ax, x[1]..x[end], y[1]..y[end], z[1]..z[end], V; algorithm=:iso, isovalue=0.0, colormap=:broc)
    
    display(fig)  # opens rotatable 3D window
    return fig
end

end # module