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
    "--ndeg"
        help = "intruder noise standard dev in degrees"
        arg_type = Float64
        default = 10.0
    "filename"
        required = true
end

args = ArgParse.parse_args(s)

phi = FEATURES

ndeg_arg = args["ndeg"]

INTRUDER.heading_std = ndeg_arg*pi/180.0

cfnames = {10.0 => "../data/box_10k_collisions.ic",
            5.0 => "../data/box_10k_collisions_5.ic",
            15.0 => "../data/box_10k_collisions_15.ic"}
mfnames = {10.0 => "../data/box_10k_mixed.ic",
            5.0 => "../data/box_10k_mixed_5.ic",
            15.0 => "../data/box_10k_mixed_5.ic"}

@show c_ic_fname = cfnames[ndeg_arg]
@show m_ic_fname = mfnames[ndeg_arg]

# c_ic_fname = "../data/10k_collisions.ic"
col_data = JLD.load(c_ic_fname)

col_ics = col_data["ics"]
col_seeds = col_data["seeds"]

# m_ic_fname = "../data/10k_mixed.ic"
mixed_data = JLD.load(m_ic_fname)

mixed_ics = mixed_data["ics"]
mixed_seeds = mixed_data["seeds"]
num_mixed_collisions = mixed_data["num_collisions"]

data = JLD.load(args["filename"])

policies = data["policies"]
risk_ratios = Array(Float64, length(policies))
deviations = Array(Int64, length(policies))
avg_delays = Array(Float64, length(policies))
avg_delays_all = Array(Float64, length(policies))

baseline_completion_time = 31

prefs=Array(Any,length(policies))

# i = 1
for i in 1:length(policies)
    tic()

    policy = policies[i]

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

    @show risk_ratio = n_nmac/length(col_ics)
    @show deviations[i]
    @show avg_delays_all[i] 
    risk_ratios[i] = risk_ratio
    toc()

end

input_filename = args["filename"]
@show filename = "../data/re_eval_$(int(ndeg_arg))_$(Dates.format(Dates.now(),"u-d_HHMM")).jld"
JLD.@save filename risk_ratios policies deviations avg_delays baseline_completion_time avg_delays_all args SIM INTRUDER OWNSHIP input_filename

