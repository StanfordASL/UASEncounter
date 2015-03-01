
using EncounterModel
@everywhere using EncounterSimulation
import Dates
import HDF5, JLD


# evaluate four policies
# g greedy trl
# c conservative trl
# o optimized trl
# t optimized turning

rng = MersenneTwister(0)

# load policies
pol_data = JLD.load("../data/first_pol.jld")
g_pol = ConstPolicy(HeadingHRL(SIM.legal_D))
c_pol = ConstPolicy(HeadingHRL(2.0*SIM.legal_D))
o_pol = extract_from_record(pol_data["hrl_policy"])
t_pol = extract_from_record(pol_data["accel_policy"])

@show pol_data["hrl_policy"].phi_desc
@show pol_data["accel_policy"].phi_desc

# generate tests
num_tests = 10000

tests=EncounterTest[]

println("creating tests...")
for i=1:num_tests
    state = gen_init_state(rng)
    push!(tests, EncounterTest(EncounterTestInputData(state,
                                                      policy=g_pol,
                                                      seed=i)))
    push!(tests, EncounterTest(EncounterTestInputData(state,
                                                      policy=c_pol,
                                                      seed=i)))
    push!(tests, EncounterTest(EncounterTestInputData(state,
                                                      policy=o_pol,
                                                      seed=i)))
    push!(tests, EncounterTest(EncounterTestInputData(state,
                                                      policy=t_pol,
                                                      seed=i)))
end

# run tests
try
    tic()
    run!(tests, parallel=true, store_hist=false)
    toc()
catch e
    run(`sdo false`)
    rethrow(e)
end

# save results
println("saving results...")
tic()
# JLD.save("../data/policy_tests_$(Dates.now()).jld", "test_records", [make_record(t) for t in tests])
JLD.save("/mnt/data/zach_policy_tests_$(Dates.now()).jld", "test_records", [make_record(t) for t in tests])
toc()

