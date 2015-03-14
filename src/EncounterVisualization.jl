module EncounterVisualization

# import Gadfly
using PyPlot
using EncounterModel
using EncounterFeatures
using EncounterSimulation
# using PGFPlots

export plot_value_grid

function plot_value_grid(phi::FeatureBlock, theta::Vector{Float64}, is::IntruderState, ownship_heading::Float64, n=100;deviated=false)
    ymin = -600.0
    ymax = 600.0
    xmin = -100.0
    xmax = 1100.0
    xpoints = linspace(xmin, xmax, n)
    ypoints = linspace(ymin, ymax, n)
    vals = Array(Float64, n, n)
    for i in 1:n
        for j in 1:n
            state = EncounterState([xpoints[i], ypoints[j], ownship_heading], is, false,deviated)
            vals[i, j] = sum(evaluate(phi,state)'*theta)
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

function plot_policy_grid(policy::EncounterPolicy, is::IntruderState, ownship_heading::Float64, n=100; threshold=-1.0)
    ymin = -600.0
    ymax = 600.0
    xmin = -100.0
    xmax = 1100.0
    extent = (ymin, ymax, xmin, xmax)
    xpoints = linspace(xmin, xmax, n)
    ypoints = linspace(ymin, ymax, n)
    vals = Array(Float64, n, n)
    any_negligible = false
    for i in 1:n
        for j in 1:n
            state = EncounterState([xpoints[i], ypoints[j], ownship_heading], is, false)
            if typeof(policy)==LinearQValuePolicy
                qs=Array(Float64, length(policy.actions))
                for k in 1:length(qs)
                    qs[k] = sum(evaluate(policy.phi, state)'*policy.thetas[k])
                end
                if maximum(qs) - minimum(qs) <= threshold
                    vals[i,j] = 1
                    any_negligible = true
                else
                    vals[i,j] = indmax(qs) +1
                end
            else
                vals[i,j] = query_policy_ind(policy, state)+1
            end
            # vals[i,j] = maximum(qs)-minimum(qs)
        end
    end
    # @show vals
    clf()
    imshow(vals, origin="lower", extent=extent, interpolation="nearest", cmap=PyPlot.cm["gist_rainbow"])
    xlabel("y (east)")
    ylabel("x (north)")
    # colorbar()
    cbar = colorbar(ticks=[1:length(policy.actions)+1])
    if any_negligible
        cbar[:ax][:set_yticklabels](["Negligible Difference", [string(a) for a in policy.actions]])
    else
        cbar[:ax][:set_yticklabels]([string(a) for a in policy.actions])
    end
    ax=gca()
    idx = 50*cos(is[3])
    idy = 50*sin(is[3])
    ax[:arrow](is[2], is[1], idy, idx, head_width=20, head_length=20)
    odx = 50*cos(ownship_heading)
    ody = 50*sin(ownship_heading)
    ax[:arrow](0,0, ody,odx, head_width=20, head_length=20)
end

end
