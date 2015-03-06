@everywhere areload()

@everywhere using EncounterModel: IntruderParams, OwnshipParams, SimParams, EncounterState, ownship_control, ownship_dynamics, encounter_dynamics, next_state_from_pd, post_decision_state, reward, SIM
# @everywhere using WHack
@everywhere using HRL: heading_hrl
@everywhere using EncounterFeatures
@everywhere using GridInterpolations: SimplexGrid, RectangleGrid
@everywhere using EncounterValueIteration: run_sims, iterate
import ProfileView

# using Debug

@everywhere begin
    goal_dist_points = linspace(0.0, 500.0, 10)
    goal_bearing_points = linspace(0.0, 2*pi, 15)

    # intruder_dist_points = [0.0, linspace(120.0,700.0,11).^2/700.0]
    intruder_dist_points = linspace(0.0, 700.0, 12) 
    intruder_bearing_points = linspace(-pi/2, pi/2, 12)
    intruder_heading_points = linspace(0.0, 2*pi, 12)

    features = [
        :f_in_goal,
        # f_abs_goal_bearing,
        :f_goal_dist,
        :f_one,
        ParameterizedFeatureFunction(:f_radial_goal_grid, SimplexGrid(goal_dist_points, goal_bearing_points)),
        ParameterizedFeatureFunction(:f_focused_intruder_grid, SimplexGrid(intruder_dist_points, intruder_bearing_points, intruder_heading_points)),
        :f_conflict,
        # f_intruder_dist,
    ]
    # features = f_radial_intruder_grid
    phi = FeatureBlock(features)
    NEV = 20
end

rng0 = MersenneTwister(0)

@everywhere const lD = SIM.legal_D
@everywhere const ACTIONS = [HeadingHRL(D) for D in [lD, 1.1*lD, 1.2*lD, 1.5*lD, 2.0*lD]]

sims_per_policy = 1000
theta = zeros(phi.length)
theta_new = iterate(phi, theta, ACTIONS, 1000, convert_to_sparse=true)
Profile.clear()

@profile for i in 1:1
    println("starting policy iteration $i ($sims_per_policy simulations)")
    theta_new = iterate(phi, theta, ACTIONS, sims_per_policy, rng_seed_offset=i*1120000+1, convert_to_sparse=true)
    println("max difference: $(norm(theta_new - theta, Inf))")
    println("2-norm difference: $(norm(theta_new - theta))")
    theta = theta_new
end

ProfileView.view()
