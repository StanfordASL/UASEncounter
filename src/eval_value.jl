
using EncounterModel
@everywhere using EncounterSimulation
import EncounterVisualization
using GridInterpolations


n = 100
ymin = -600.0
ymax = 600.0
xmin = -100.0
xmax = 1100.0
xpoints = linspace(xmin, xmax, n)
ypoints = linspace(ymin, ymax, n)
vals = Array(Float64, n, n)

@everywhere function run10(test)
    m = 10
    reward = 0;
    for i in 1:m
        run!(test, announce=false)
        reward += test.output.reward
    end
    return reward/m
end

try
    is = [550.0, -300, pi/180.0*135.0] # out of the way
    ownship_heading = 0.0

    refs = Array(Any, n, n)
    for i in 1:n
        for j in 1:n
            state = EncounterState([xpoints[i], ypoints[j], ownship_heading], is, false)
            test = EncounterTest(state)
            test.input.policy = ConstPolicy(HeadingHRL(1.2*SIM.legal_D))
            # refs[i, j] = @spawn run!(test, announce=false)
            refs[i, j] = @spawn run10(test)
        end
    end
    println("tests spawned")
    for i in 1:n
        for j in 1:n
            print("\rwaiting for [$i,$j]")
            # test = fetch(refs[i,j])
            # vals[i, j] = test.output.reward
            vals[i,j] = fetch(refs[i,j])
        end
    end
    EncounterVisualization.plot_value_grid(vals, extent=(ymin, ymax, xmin, xmax))

    xgp = linspace(xmin, xmax, 50)
    ygp = linspace(ymin, ymax, 50)
    grid = RectangleGrid(xgp, ygp)

    phi = zeros(Float64, n*n, length(grid))

    for i in 1:n
        for j in 1:n
            ind, val = interpolants(grid, [xpoints[i], ypoints[j]])
            phi[(i-1)*n+j,ind] = val
        end
    end

    theta = pinv(phi)*vec(vals')

    gridvals = Array(Float64, size(vals))
    for i in 1:n
        for j in 1:n
            gridvals[i, j] = interpolate(grid, theta, [xpoints[i], ypoints[j]])
        end
    end

    EncounterVisualization.plot_value_grid(gridvals, extent=(ymin, ymax, xmin, xmax))

catch e
    run(`sdo false`)
    rethrow(e)
end


run(`sdo true`)
