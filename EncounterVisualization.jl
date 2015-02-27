module EncounterVisualization

# import Gadfly
using PyPlot
using EncounterModel: IntruderState, EncounterState
using EncounterFeatures: AssembledFeatureBlock

export plot_value_grid

function plot_value_grid(phi::AssembledFeatureBlock, lambda::Vector{Float64}, is::IntruderState, ownship_heading::Float64, n=100)
    ymin = -600.0
    ymax = 600.0
    xmin = -100.0
    xmax = 1100.0
    xpoints = linspace(xmin, xmax, n)
    ypoints = linspace(ymin, ymax, n)
    vals = Array(Float64, n, n)
    for i in 1:n
        for j in 1:n
            state = EncounterState([xpoints[i], ypoints[j], ownship_heading], is, false)
            vals[i, j] = sum(phi.features(state)'*lambda)
        end
    end
    plot_value_grid(vals, extent=(ymin, ymax, xmin, xmax))
    ax=gca()
    idx = 50*cos(is[3])
    idy = 50*sin(is[3])
    ax[:arrow](is[2], is[1], idy, idx, head_width=20, head_length=20)
    odx = 50*cos(ownship_heading)
    ody = 50*sin(ownship_heading)
    ax[:arrow](0,0, ody,odx, head_width=20, head_length=20)
    # Gadfly.spy(grid)
end

function plot_value_grid(vals::Array{Float64,2}; extent=(-600.0, 600.0, -100.0, 1100.0))
    clf()
    imshow(vals, origin="lower", extent=extent, interpolation="nearest", cmap=PyPlot.cm["gist_rainbow"])
    xlabel("y (east)")
    ylabel("x (north)")
    colorbar()
end


end
