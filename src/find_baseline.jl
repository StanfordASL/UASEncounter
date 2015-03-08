using EncounterSimulation
using EncounterModel

ic = EncounterState([0,0,0],[10000,10000,0],false,false)
policy = ConstPolicy(BankControl(0.0))

t=EncounterTest(EncounterTestInputData(ic,policy=policy))
run!(t)
t.output.steps_before_end
