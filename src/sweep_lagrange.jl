@everywhere using EncounterModel
@everywhere using EncounterFeatures
@everywhere using GridInterpolations
@everywhere using EncounterValueIteration
@everywhere using EncounterSimulation

import ArgParse
import HDF5, JLD
import Dates

s = ArgParse.ArgParseSettings()

ArgParse.@add_arg_table s begin
    "-a"
        help = "\"trl\" or \"turning\" actions"
        arg_type = ASCIIString
        default = "turning"
end

args = ArgParse.parse_args(s)

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

a_arg = args["a"]
@show filename = "../data/$(a_arg)_lagrange_sweep_$(Dates.format(Dates.now(),"u-d_HHMM")).jld"
if a_arg == "turning"
    @show actions = EncounterAction[BankControl(b) for b in [-OWNSHIP.max_phi, -OWNSHIP.max_phi/2, 0.0, OWNSHIP.max_phi/2, OWNSHIP.max_phi]]
elseif a_arg == "trl"
    lD = SIM.legal_D
    @show actions = EncounterAction[HeadingHRL(D) for D in [1.5*lD, 2.0*lD, 3.0*lD, 5.0*lD, 10.0*lD]]
else
    error("Invalid -a input. Expected \"trl\" or \"turning\"; got \"$a_arg\"")
end


lambdas = logspace(3,6,8)

c_ic_fname = "../data/10k_collisions.ic"
col_data = JLD.load(c_ic_fname)

col_ics = col_data["ics"]
col_seeds = col_data["seeds"]

m_ic_fname = "../data/10k_mixed.ic"
mixed_data = JLD.load(m_ic_fname)

mixed_ics = mixed_data["ics"]
mixed_seeds = mixed_data["seeds"]
num_mixed_collisions = mixed_data["num_collisions"]

risk_ratios = Array(Float64, length(lambdas))
policies = Array(Any, length(lambdas))
deviations = Array(Int64, length(lambdas))
avg_delays = Array(Float64, length(lambdas))

baseline_completion_time = 61

# i = 1
for i in 1:length(lambdas)
    tic()
    lambda = lambdas[i]
    rm = DeviationAndTimeReward(100, 1, 100, lambda)
    policy = find_policy(phi, rm, actions, intruder_grid)
    policies[i] = policy

    col_tests = test_policy(policy, col_ics, col_seeds)   
    n_nmac = sum([t.output.nmac for t in col_tests])
    # n_dev = sum([t.output.deviated for t in col_tests])
    # deviation_tests = filter(t->t.output.deviated, tests)

    mixed_tests = test_policy(policy, mixed_ics, mixed_seeds)
    deviations[i] = sum([t.output.deviated for t in mixed_tests])
    dev_tests = filter(t->t.output.deviated, mixed_tests)
    @show [t.output.steps_before_end-baseline_completion_time for t in dev_tests]
    avg_delays[i] = mean([t.output.steps_before_end-baseline_completion_time for t in dev_tests])

    @show lambda
    @show risk_ratio = n_nmac/length(col_ics)
    @show deviations[i]
    @show avg_delays[i] 
    risk_ratios[i] = risk_ratio
    toc()
end

JLD.@save filename lambdas risk_ratios policies deviations avg_delays baseline_completion_time
