import HDF5, JLD
using EncounterModel
using EncounterSimulation
import Dates

import ArgParse

s = ArgParse.ArgParseSettings()

ArgParse.@add_arg_table s begin

    "--ndeg"
        help = "intruder noise standard dev in degrees"
        arg_type = Float64
        default = 10.0
end

args = ArgParse.parse_args(s)


@show filename = "../data/D_sweep_$(int(args["ndeg"]))_$(Dates.format(Dates.now(),"u-d_HHMM")).jld"

cfnames = {10.0 => "../data/box_10k_collisions.ic",
            5.0 => "../data/box_10k_collisions_5.ic",
            15.0 => "../data/box_10k_collisions_15.ic"}
mfnames = {10.0 => "../data/box_10k_mixed.ic",
            5.0 => "../data/box_10k_mixed_5.ic",
            15.0 => "../data/box_10k_mixed_5.ic"}

INTRUDER.heading_std=args["ndeg"]*pi/180.0

@show ic_filename = cfnames[args["ndeg"]]
ic_data = JLD.load(ic_filename)

@show m_ic_fname = mfnames[args["ndeg"]]
mixed_data = JLD.load(m_ic_fname)

mixed_ics = mixed_data["ics"]
mixed_seeds = mixed_data["seeds"]
num_mixed_collisions = mixed_data["num_collisions"]

ics = ic_data["ics"]
seeds = ic_data["seeds"]

target_rr = 0.05
sims_per_iter = 10000

risk_ratios = Float64[]
deviations = Float64[]
avg_delays = Float64[]
avg_delays_all = Float64[]
baseline_completion_time = 31

# Ds = logspace(2,4,8)
nDs = {10.0=>[250, 300, 350, 400, 500],
        5.0=>[170, 200, 250, 300, 400]}
Ds = nDs[args["ndeg"]]

for i in 1:length(Ds)
    println("===================")
    @show D = Ds[i]
    policy = ConstPolicy(HeadingHRL(D))

    ts = test_policy(policy, ics, seeds)
    n_nmac = sum([t.output.nmac for t in ts])
    @show risk_ratio = n_nmac/sims_per_iter
    push!(risk_ratios, risk_ratio)

    mixed_tests = test_policy(policy, mixed_ics, mixed_seeds)
    @show devs = sum([t.output.deviated for t in mixed_tests])
    push!(deviations, devs)
    dev_no_nmac(t) = t.output.deviated && !t.output.nmac
    dev_tests = filter(dev_no_nmac, mixed_tests)
    # @show [t.output.steps_before_end-baseline_completion_time for t in dev_tests]
    @show avg_delay = mean([t.output.steps_before_end-baseline_completion_time for t in dev_tests])
    @show avg_delay_all = mean([t.output.steps_before_end-baseline_completion_time for t in mixed_tests])
    push!(avg_delays, avg_delay)
    push!(avg_delays_all, avg_delay_all)
end

JLD.@save filename Ds risk_ratios deviations avg_delays baseline_completion_time avg_delays_all
