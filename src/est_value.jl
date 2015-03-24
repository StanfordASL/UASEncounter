using EncounterModel
@everywhere using EncounterFeatures
using GridInterpolations
@everywhere using EncounterValueIteration
import EncounterVisualization
import HDF5, JLD
import Dates

phi = FEATURES

# @show phi.description
# @show file_prefix = "no_ripple_30k"
@show file_prefix = "cdc_val_fig"

lD = SIM.legal_D
@show actions = EncounterAction[HeadingHRL(D) for D in [lD, 1.5*lD, 2.0*lD, 3.0*lD, 4.0*lD]]
# actions = EncounterAction[BankControl(b) for b in [-OWNSHIP.max_phi, -OWNSHIP.max_phi/2, 0.0, OWNSHIP.max_phi/2, OWNSHIP.max_phi]]

rng0 = MersenneTwister(0)

theta = zeros(length(phi))
plot_is = [550.0, -300.0, pi/180.0*135.0]
plot_heading = 0.0
rew = REWARD
@everywhere snap_generator(rng) = gen_state_snap_to_grid(rng, INTRUDER_GRID, GOAL_GRID)

rng0 = MersenneTwister(0)

theta = zeros(length(phi))

iters = 10000*ones(Int64,35)

for i in 1:length(iters)
    tic()
    ic_batch = [gen_ic_batch_for_grid(rng0, INTRUDER_GRID,GOAL_GRID),
                gen_undeviated_ic_batch(rng0, INTRUDER_GRID, num=200)]
    theta_new = iterate(phi, theta, rew, actions, iters[i],
                        rng_seed_offset=2048*i,
                        state_gen=snap_generator,
                        parallel=true,
                        ic_batch=ic_batch,
                        output_prefix="\r[$i ($(iters[i]))]",
                        output_suffix="",
                        parallel=true)
    theta = theta_new

    EncounterVisualization.plot_value_grid(phi, theta, plot_is, plot_heading) 

    toc()
end

@show filename = "../data/$(file_prefix)_$(Dates.format(Dates.now(),"u-d_HHMM")).value"

JLD.@save filename phi theta 

# theta = find_value(phi, rew, actions, INTRUDER_GRID, GOAL_GRID, parallel=true, iters=50000*ones(Int64,30))

readline(STDIN)

# try
#     rm("../data/current.value")
# catch e
#     println(e)
#     println("continuing anyways...")
# end
# 
# symlink(filename, "../data/current.value")
