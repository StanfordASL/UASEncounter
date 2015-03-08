@everywhere using EncounterModel
@everywhere using EncounterFeatures
@everywhere using GridInterpolations
@everywhere using EncounterValueIteration
@everywhere using EncounterSimulation

import HDF5, JLD

@everywhere begin
    goal_dist_points = linspace(0.0, 500.0, 10)
    goal_bearing_points = linspace(0.0, 2*pi, 15)

    intruder_dist_points = linspace(0.0, 700.0, 12) 
    intruder_bearing_points = linspace(-pi/2, pi/2, 12)
    intruder_heading_points = linspace(0.0, 2*pi, 12)
    intruder_grid = RectangleGrid(intruder_dist_points, intruder_bearing_points, intruder_heading_points)

    features = [
        :f_in_goal,
        :f_goal_dist,
        :f_one,
        :f_has_deviated,
        ParameterizedFeatureFunction(:f_radial_goal_grid, RectangleGrid(goal_dist_points, goal_bearing_points), true),
        ParameterizedFeatureFunction(:f_focused_intruder_grid, intruder_grid, true),
        :f_conflict,
    ]
    phi = FeatureBlock(features)
end

@show filename = "../data/first_turning_lagrange_sweep.jld"
actions = EncounterAction[BankControl(b) for b in [-OWNSHIP.max_phi, -OWNSHIP.max_phi/2, 0.0, OWNSHIP.max_phi/2, OWNSHIP.max_phi]]

lambdas = logspace(3,6,5)

ic_filename = "../data/10k_collisions.ic"
ic_data = JLD.load(ic_filename)

ics = ic_data["ics"]
seeds = ic_data["seeds"]

risk_ratios = Array(Float64, length(lambdas))
policies = Array(Any, length(lambdas))

for i in 1:length(lambdas)
    tic()
    lambda = lambdas[i]
    rm = DeviationAndTimeReward(100, 1, 100, lambda)
    policy = find_policy(phi, rm, actions, intruder_grid)
    policies[i] = policy

    tests = test_policy(policy, ics, seeds)   
    n_nmac = sum([t.output.nmac for t in tests])
    @show n_dev = sum([t.output.deviated for t in tests])
    # deviation_tests = filter(t->t.output.deviated, tests)

    @show lambda
    @show risk_ratio = n_nmac/length(ics)
    risk_ratios[i] = risk_ratio
    toc()
end

JLD.@save filename lambdas risk_ratios policies
