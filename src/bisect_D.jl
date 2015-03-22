import HDF5, JLD
using EncounterModel
using EncounterSimulation
import Dates

@show filename = "../data/D_bisection_$(Dates.format(Dates.now(),"u-d_HHMM")).jld"

ic_filename = "../data/box_10k_collisions.ic"
ic_data = JLD.load(ic_filename)

m_ic_fname = "../data/box_10k_mixed.ic"
mixed_data = JLD.load(m_ic_fname)

mixed_ics = mixed_data["ics"]
mixed_seeds = mixed_data["seeds"]
num_mixed_collisions = mixed_data["num_collisions"]

ics = ic_data["ics"]
seeds = ic_data["seeds"]

max_D = 1000.0
min_D = 0.0
target_rr = 0.05
sims_per_iter = 10000

bound = 500.0
Ds = Float64[]
risk_ratios = Float64[]
deviations = Float64[]
avg_delays = Float64[]
avg_delays_all = Float64[]
baseline_completion_time = 31

while max_D - min_D > 1.0
    println("===================")
    @show new_D = (max_D + min_D)/2.0
    # policy = ConstPolicy(HeadingHRL(new_D))
    policy = ConstPolicy(BoundedHeadingHRL(new_D,bound))
    # n_nmac = 0
    # ts = Array(EncounterTest, sims_per_iter)
    # for i in 1:sims_per_iter
    #     ts[i] = EncounterTest(EncounterTestInputData(ics[i], policy=policy, seed=seeds[i]))
    #     # run!(t, store_hist=false)
    #     # if t.output.nmac
    #     #     n_nmac += 1
    #     # end
    #     # print("\rcompleted simulation $i")
    # end
    # print("\n")
    # run!(ts, store_hist=false, parallel=true)

    ts = test_policy(policy, ics, seeds)
    n_nmac = sum([t.output.nmac for t in ts])
    @show risk_ratio = n_nmac/sims_per_iter
    push!(Ds, new_D)
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

    if risk_ratio > target_rr
        @show min_D = new_D
    else
        @show max_D = new_D
    end
end

JLD.@save filename Ds risk_ratios deviations avg_delays baseline_completion_time avg_delays_all bound
