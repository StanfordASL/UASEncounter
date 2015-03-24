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
    "--Qvalue"
        help = "use Q-value policies"
        action = :store_true
    "--ndeg"
        help = "intruder noise standard dev in degrees"
        arg_type = Float64
        default = 10.0
end

args = ArgParse.parse_args(s)

phi = FEATURES

a_arg = args["a"]
ndeg_arg = args["ndeg"]

INTRUDER.heading_std = ndeg_arg*pi/180.0

iters=[10000*ones(Int64,34),50000]
if a_arg == "turning"
    @show actions = EncounterAction[BankControl(b) for b in [-OWNSHIP.max_phi, -OWNSHIP.max_phi/2, 0.0, OWNSHIP.max_phi/2, OWNSHIP.max_phi]]
    # lambdas = logspace(3,7,6)
    # lambdas = logspace(1,5,8)
    # lambdas = logspace(2,4,4)
    # lambdas = [100, 200, 400, 550, 700]
    lambdas = [300, 500, 700, 1000, 1500]
    # iters=[10000*ones(Int64,59),50000]
    iters = 50000*ones(Int64,35)
elseif a_arg == "turningpick"
    @show actions = EncounterAction[BankControl(b) for b in [-OWNSHIP.max_phi, -OWNSHIP.max_phi/2, 0.0, OWNSHIP.max_phi/2, OWNSHIP.max_phi]]
    lambdas = [1500]
    iters = 50000*ones(Int64,35)
elseif a_arg == "trl"
    lD = SIM.legal_D
    # @show actions = EncounterAction[HeadingHRL(D) for D in [lD, 1.5*lD, 2.0*lD, 2.5*lD, 3.0*lD]]
    @show actions = EncounterAction[HeadingHRL(D) for D in [lD, 1.2*lD, 1.5*lD, 2.0*lD]]
    lambdas = logspace(1.8,4,5)
elseif a_arg == "trlmatch"
    lD = SIM.legal_D
    @show actions = EncounterAction[HeadingHRL(D) for D in [lD, 1.5*lD, 2.0*lD, 2.3*lD]]
    lambdas = logspace(1,5,8)
elseif a_arg == "trlbox"
    lD = SIM.legal_D
    # @show actions = EncounterAction[HeadingHRL(D) for D in [lD, 1.2*lD, 1.4*lD, 1.6*lD, 2.0*lD]]
    @show actions = EncounterAction[HeadingHRL(D) for D in [lD, 1.2*lD, 1.5*lD, 2.0*lD, 2.5*lD]]
    lambdas = logspace(1,5,8)
elseif a_arg == "trlcons"
    lD = SIM.legal_D
    @show actions = EncounterAction[HeadingHRL(D) for D in [lD, 1.5*lD, 2.0*lD, 3.0*lD, 4.0*lD]]
    lambdas = logspace(2,5,7)
elseif a_arg == "trllong"
    lD = SIM.legal_D
    @show actions = EncounterAction[HeadingHRL(D) for D in [lD, 1.5*lD, 2.0*lD, 3.0*lD, 4.0*lD]]
    lambdas = [100]
    iters = 50000*ones(Int64,35)
elseif a_arg == "trlzero"
    lD = SIM.legal_D
    @show actions = EncounterAction[HeadingHRL(D) for D in [0.0, lD, 1.5*lD, 2.0*lD, 3.0*lD, 4.0*lD]]
    lambdas = logspace(2,4,6)
elseif a_arg == "trlinsane"
    lD = SIM.legal_D
    @show actions = EncounterAction[HeadingHRL(D) for D in [0.0, lD, 1.5*lD, 2.0*lD, 3.0*lD, 4.0*lD]]
    lambdas = [0]
elseif a_arg == "trlbounded"
    lD = SIM.legal_D
    actions = EncounterAction[HeadingHRL(D) for D in [0.0, lD, 1.5*lD, 2.0*lD, 3.0*lD, 4.0*lD]]
    actions[1] = BoundedHeadingHRL(lD, 500)
    @show actions
    lambdas = logspace(2,4,6)
else
    error("Invalid -a input. Expected \"trl\" or \"turning\"; got \"$a_arg\"")
end

cfnames = {10.0 => "../data/box_10k_collisions.ic",
            5.0 => "../data/box_10k_collisions_5.ic",
            15.0 => "../data/box_10k_collisions_15.ic"}
mfnames = {10.0 => "../data/box_10k_mixed.ic",
            5.0 => "../data/box_10k_mixed_5.ic",
            15.0 => "../data/box_10k_mixed_5.ic"}

c_ic_fname = cfnames[ndeg_arg]
m_ic_fname = mfnames[ndeg_arg]

# c_ic_fname = "../data/10k_collisions.ic"
col_data = JLD.load(c_ic_fname)

col_ics = col_data["ics"]
col_seeds = col_data["seeds"]

# m_ic_fname = "../data/10k_mixed.ic"
mixed_data = JLD.load(m_ic_fname)

mixed_ics = mixed_data["ics"]
mixed_seeds = mixed_data["seeds"]
num_mixed_collisions = mixed_data["num_collisions"]

risk_ratios = Array(Float64, length(lambdas))
policies = Array(Any, length(lambdas))
deviations = Array(Int64, length(lambdas))
avg_delays = Array(Float64, length(lambdas))
avg_delays_all = Array(Float64, length(lambdas))
rms = Array(Any, length(lambdas))

baseline_completion_time = 31

prefs=Array(Any,length(lambdas))

# i = 1
for i in 1:length(lambdas)
    tic()
    lambda = lambdas[i]
    # rm = DeviationAndTimeReward(0, 1, 100, lambda)
    rm = deepcopy(REWARD)
    rm.nmac_lambda = lambda
    rms[i] = rm

# ======== FINE
    policy = find_policy(phi, rm, actions, INTRUDER_GRID, GOAL_GRID, post_decision=!args["Qvalue"], parallel=true, iters=iters)
# ========

# ======== COARSE
#     prefs[i] = @spawn find_policy(phi, rm, actions, INTRUDER_GRID, GOAL_GRID, post_decision=!args["Qvalue"], parallel=false, num_short=27, num_long=3)
#     sleep(1)
# end
# 
# # println("========================")
# # println("Done spawning policies!")
# # println("========================")
# 
# for i in 1:length(lambdas)
#     lambda=lambdas[i]
#     policy=fetch(prefs[i])
# ========

    policies[i] = policy

    col_tests = test_policy(policy, col_ics, col_seeds)   
    n_nmac = sum([t.output.nmac for t in col_tests])
    # n_dev = sum([t.output.deviated for t in col_tests])
    # deviation_tests = filter(t->t.output.deviated, tests)

    mixed_tests = test_policy(policy, mixed_ics, mixed_seeds)

    deviations[i] = sum([t.output.deviated for t in mixed_tests])
    dev_no_nmac(t) = t.output.deviated && !t.output.nmac
    dev_tests = filter(dev_no_nmac, mixed_tests)
    # @show [t.output.steps_before_end-baseline_completion_time for t in dev_tests]
    if length(dev_tests) > 0
        avg_delays[i] = mean([t.output.steps_before_end-baseline_completion_time for t in dev_tests])
    else
        avg_delays[i] = 0.0
    end
    avg_delays_all[i] = mean([t.output.steps_before_end-baseline_completion_time for t in mixed_tests])

    @show lambda
    @show risk_ratio = n_nmac/length(col_ics)
    @show deviations[i]
    @show avg_delays_all[i] 
    risk_ratios[i] = risk_ratio
    toc()

    @show filename = "../data/$(a_arg)_sweep_$(int(ndeg_arg))_$(Dates.format(Dates.now(),"u-d_HHMM")).jld"
    JLD.@save filename lambdas risk_ratios policies deviations avg_delays baseline_completion_time avg_delays_all args rms SIM INTRUDER OWNSHIP
end
