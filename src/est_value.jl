@everywhere using EncounterModel: IntruderParams, OwnshipParams, SimParams, EncounterState, EncounterAction, HeadingHRL, BankControl, ownship_control, ownship_dynamics, encounter_dynamics, next_state_from_pd, post_decision_state, reward, SIM, OWNSHIP
# @everywhere using WHack
@everywhere using EncounterFeatures
@everywhere using GridInterpolations
@everywhere using EncounterValueIteration
# import EncounterVisualization
import SVDSHack
import HDF5, JLD
import Dates

# using Debug

@everywhere begin
    goal_dist_points = linspace(0.0, 500.0, 10)
    goal_bearing_points = linspace(0.0, 2*pi, 15)

    # intruder_dist_points = [0.0, linspace(120.0,700.0,11).^2/700.0]
    intruder_dist_points = linspace(0.0, 700.0, 12) 
    intruder_bearing_points = linspace(-pi/2, pi/2, 12)
    intruder_heading_points = linspace(0.0, 2*pi, 12)
    intruder_grid = RectangleGrid(intruder_dist_points, intruder_bearing_points, intruder_heading_points)
#     intruder_dist_points = linspace(0.0, 700.0, 16) 
#     intruder_bearing_points = linspace(-pi/2, pi/2, 16)
#     intruder_heading_points = linspace(0.0, 2*pi, 12)
    half_intruder_grid_param = [:num_intruder_dist=>12,
                                :max_intruder_dist=>700.0,
                                :num_intruder_bearing=>12,
                                :num_intruder_heading=>12]

    features = [
        :f_in_goal,
        # f_abs_goal_bearing,
        :f_goal_dist,
        :f_one,
        ParameterizedFeatureFunction(:f_radial_goal_grid, RectangleGrid(goal_dist_points, goal_bearing_points), true),
        ParameterizedFeatureFunction(:f_focused_intruder_grid, intruder_grid, true),
        # ParameterizedFeatureFunction(f_half_intruder_bin_grid, half_intruder_grid_param, true),
        :f_conflict,
        # f_intruder_dist,
    ]
    # features = f_radial_intruder_grid
    phi = FeatureBlock(features)
    NEV = 20
end

# @show phi.description
@show file_prefix = "trl_end_50k"

@everywhere const lD = SIM.legal_D
@everywhere const actions = EncounterAction[HeadingHRL(D) for D in [lD, 2.0*lD, 3.0*lD, 5.0*lD, 10.0*lD]]
# @everywhere const actions = EncounterAction[BankControl(b) for b in [-OWNSHIP.max_phi, -OWNSHIP.max_phi/2, 0.0, OWNSHIP.max_phi/2, OWNSHIP.max_phi]]

rng0 = MersenneTwister(0)

theta = zeros(length(phi))
plot_is = [550.0, -300.0, pi/180.0*135.0]
plot_heading = 0.0
@everywhere snap_generator(rng) = gen_state_snap_to_grid(rng, intruder_grid)
try
    for i in 1:30
        sims_per_policy = 10000
        println("starting value iteration $i ($sims_per_policy simulations)")
        ic_batch = gen_ic_batch_for_grid(rng0, intruder_grid)
        theta_new = iterate(phi, theta, actions, sims_per_policy, rng_seed_offset=i*1120000+1, state_gen=snap_generator, parallel=true, ic_batch=ic_batch)
        theta = theta_new
    end

    sims_per_policy = 50000
    println("starting final value iteration ($sims_per_policy simulations)")
    ic_batch = gen_ic_batch_for_grid(rng0, intruder_grid)
    theta_new = iterate(phi, theta, actions, sims_per_policy, rng_seed_offset=0, state_gen=snap_generator, parallel=true, ic_batch=ic_batch)
    theta = theta_new

    JLD.save("../data/$(file_prefix)_value_$(Dates.now()).jld",
             "theta", theta,
             "phi", phi,
             "actions", actions,
             "intruder_grid", intruder_grid
             )
catch e
    # JLD.save("../data/ERROR_$(file_prefix)_value_$(Dates.now()).jld", "theta", theta, "phi_description", phi.description)
    # run(`sdo false`)
    rethrow(e)
end

# run(`sdo true`)
