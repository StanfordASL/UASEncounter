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

c_ic_fname = "../data/10k_collisions.ic"
col_data = JLD.load(c_ic_fname)

col_ics = col_data["ics"]
col_seeds = col_data["seeds"]

m_ic_fname = "../data/10k_mixed.ic"
mixed_data = JLD.load(m_ic_fname)

mixed_ics = mixed_data["ics"]
mixed_seeds = mixed_seeds["seeds"]
num_mixed_collisions = mixed_seeds["num_collisions"]

risk_ratios = Array(Float64, length(lambdas))
policies = Array(Any, length(lambdas))
deviations = Array(Int64, length(lambdas))
avg_delays = Array(Int64, length(lambdas))

baseline_completion_time = 61

for i in 1:length(lambdas)
    tic()
    lambda = lambdas[i]
    rm = DeviationAndTimeReward(100, 1, 100, lambda)
    policy = find_policy(phi, rm, actions, intruder_grid)
    policies[i] = policy

    col_tests = test_policy(policy, col_ics, col_seeds)   
    n_nmac = sum([t.output.nmac for t in col_tests])
    @show n_dev = sum([t.output.deviated for t in col_tests])
    # deviation_tests = filter(t->t.output.deviated, tests)

    mixed_tests = test_policy(policy, mixed_ics, mixed_seeds)
    deviations[i] = sum([t.output.deviated for t in mixed_tests])
    dev_tests = filter(t->t.output.deviated, mixed_tests)
    avg_delays[i] = mean([t.output.steps_before_end-baseline_completion_time for t in dev_tests])

    @show lambda
    @show risk_ratio = n_nmac/length(col_ics)
    @show deviations[i]
    @show avg_delays[i] 
    risk_ratios[i] = risk_ratio
    toc()
end

JLD.@save filename lambdas risk_ratios policies deviations avg_delays
