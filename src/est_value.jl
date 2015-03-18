@everywhere using EncounterModel
# @everywhere using WHack
@everywhere using EncounterFeatures
@everywhere using GridInterpolations
@everywhere using EncounterValueIteration
import EncounterVisualization
import HDF5, JLD
import Dates

@everywhere phi = FEATURES

# @show phi.description
@show file_prefix = "turning_test"

# @everywhere const lD = SIM.legal_D
# @everywhere const actions = EncounterAction[HeadingHRL(D) for D in [lD, 2.0*lD, 3.0*lD, 5.0*lD, 10.0*lD]]
actions = EncounterAction[BankControl(b) for b in [-OWNSHIP.max_phi, -OWNSHIP.max_phi/2, 0.0, OWNSHIP.max_phi/2, OWNSHIP.max_phi]]

rng0 = MersenneTwister(0)

theta = zeros(length(phi))
plot_is = [550.0, -300.0, pi/180.0*135.0]
plot_heading = 0.0
rew = REWARD
@everywhere snap_generator(rng) = gen_state_snap_to_grid(rng, INTRUDER_GRID, GOAL_GRID)

rng0 = MersenneTwister(0)

theta = zeros(length(phi))
snap_generator(rng) = gen_state_snap_to_grid(rng, intruder_grid, goal_grid)

iters = 50000*ones(Int64,30)

for i in 1:length(iters)
    tic()
    ic_batch = [gen_ic_batch_for_grid(rng0, intruder_grid,goal_grid),
                gen_undeviated_ic_batch(rng0, intruder_grid, num=200)]
    theta_new = iterate(phi, theta, rm, actions, iters[i],
                        rng_seed_offset=2048*i,
                        state_gen=snap_generator,
                        parallel=true,
                        ic_batch=ic_batch,
                        output_prefix="\r[$i ($(iters[i]))]",
                        output_suffix="",
                        parallel=parallel)
    theta = theta_new

    EncounterVisualization.plot_value_grid(phi, theta, plot_is, plot_heading) 

    toc()
end




# theta = find_value(phi, rew, actions, INTRUDER_GRID, GOAL_GRID, parallel=true, iters=50000*ones(Int64,30))



# try
#     rm("../data/current.value")
# catch e
#     println(e)
#     println("continuing anyways...")
# end
# 
# symlink(filename, "../data/current.value")
