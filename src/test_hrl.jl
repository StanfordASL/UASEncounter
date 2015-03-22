using EncounterModel
using EncounterSimulation
import HDF5, JLD
using Base.Test

# test test_policy consistency
D = 200.0
p=ConstPolicy(HeadingHRL(200.0))
p2=ConstPolicy(BoundedHeadingHRL(200.0,Inf))
ic_filename = "../data/box_10k_collisions.ic"
ic_data = JLD.load(ic_filename)
ics = ic_data["ics"][1:100]
seeds = ic_data["seeds"][1:100]

ts1 = test_policy(p, ics, seeds)
ts2 = test_policy(p2, ics, seeds)

@test length(ts1)==length(ts2)
for i in 1:length(ts1)
    @test ts1[i].output.reward==ts2[i].output.reward
    @test ts1[i].output.nmac==ts2[i].output.nmac
    @test ts1[i].output.deviated==ts2[i].output.deviated
    @test ts1[i].output.steps_before_end==ts2[i].output.steps_before_end
    s1 = ts1[i].output.states
    s2 = ts2[i].output.states
    @test all(s1.==s2)
end

p3 = ConstPolicy(BoundedHeadingHRL(SIM.legal_D, 0.0))
# p3 = ConstPolicy(BoundedHeadingHRL(0.0, 0.0))
ts3 = test_policy(p3, ics, seeds)

for t in ts3
    @test t.output.deviated == false
end
