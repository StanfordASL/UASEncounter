module EncounterVisualization

# import Gadfly
using PyPlot
using EncounterModel: IntruderState, EncounterState
using EncounterFeatures: AssembledFeatureBlock
using EncounterSimulation
# using PGFPlots

export plot_value_grid

function plot_value_grid(phi::AssembledFeatureBlock, theta::Vector{Float64}, is::IntruderState, ownship_heading::Float64, n=100)
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
            vals[i, j] = sum(phi.features(state)'*theta)
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

    # val(x,y) = dot(phi.features(EncounterState([x,y,ownship_heading],is,false)),theta)
    # Axis(Plots.Image(val, (ymin,ymax), (xmin,xmax)))
end

function plot_value_grid(vals::Array{Float64,2}; extent=(-600.0, 600.0, -100.0, 1100.0))
    clf()
    imshow(vals, origin="lower", extent=extent, interpolation="nearest", cmap=PyPlot.cm["gist_rainbow"])
    xlabel("y (east)")
    ylabel("x (north)")
    colorbar()
end

function plot_policy_grid(policy::LinearQValuePolicy, is::IntruderState, ownship_heading::Float64, n=100)
    ymin = -600.0
    ymax = 600.0
    xmin = -100.0
    xmax = 1100.0
    extent = (ymin, ymax, xmin, xmax)
    xpoints = linspace(xmin, xmax, n)
    ypoints = linspace(ymin, ymax, n)
    vals = Array(Float64, n, n)
    for i in 1:n
        for j in 1:n
            state = EncounterState([xpoints[i], ypoints[j], ownship_heading], is, false)
            vals[i, j] = query_policy_ind(policy, state)
        end
    end
    clf()
    imshow(vals, origin="lower", extent=extent, interpolation="nearest", cmap=PyPlot.cm["gist_rainbow"])
    xlabel("y (east)")
    ylabel("x (north)")
    cbar = colorbar(ticks=[1:length(policy.actions)])
    cbar[:ax][:set_yticklabels]([string(a) for a in policy.actions])
    ax=gca()
    idx = 50*cos(is[3])
    idy = 50*sin(is[3])
    ax[:arrow](is[2], is[1], idy, idx, head_width=20, head_length=20)
    odx = 50*cos(ownship_heading)
    ody = 50*sin(ownship_heading)
    ax[:arrow](0,0, ody,odx, head_width=20, head_length=20)
end

end
