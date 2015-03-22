using EncounterModel
using EncounterSimulation
using Base.Test
import HDF5, JLD

init = EncounterState([0,0,0], [600,-250,135*pi/180],false,false)
init2 = deepcopy(init)

p=ConstPolicy(HeadingHRL(200.0))

test1 = EncounterTest(EncounterTestInputData(init,seed=4,policy=p))
test2 = EncounterTest(EncounterTestInputData(init2,seed=4,policy=p))
@test test1.input.initial==test2.input.initial

run!(test1)
run!(test2)

@test length(test1.output.states)==length(test2.output.states)
s1 = test1.output.states
s2 = test2.output.states
@test all(s1.==s2)
a1 = test1.output.actions
a2 = test2.output.actions
@test all(a1.==a2)
@test test1.output.reward==test2.output.reward

# test test_policy consistency
D = 200.0
p=ConstPolicy(HeadingHRL(200.0))
ic_filename = "../data/box_10k_collisions.ic"
ic_data = JLD.load(ic_filename)
ics = ic_data["ics"]
seeds = ic_data["seeds"]

ts1 = test_policy(p, ics, seeds)
ts2 = test_policy(p, ics, seeds)

@test length(ts1)==length(ts2)
for i in 1:length(ts1)
    @test ts1[i].output.reward==ts2[i].output.reward
    @test ts1[i].output.nmac==ts2[i].output.nmac
    @test ts1[i].output.deviated==ts2[i].output.deviated
    @test ts1[i].output.steps_before_end==ts2[i].output.steps_before_end
    s1 = ts1[i].output.states
    s2 = ts2[i].output.states
    a1 = ts1[i].output.actions
    a2 = ts2[i].output.actions
    @test all(s1.==s2)
    @test all(a1.==a2)
end
