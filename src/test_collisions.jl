import HDF5, JLD
using EncounterModel
using EncounterSimulation
using Base.Test

col_data = JLD.load("../data/box_10k_collisions.ic")

col_ics = col_data["ics"]
col_seeds = col_data["seeds"]

p = ConstPolicy(BankControl(0.0))

for i in 1:1000
    t = EncounterTest(EncounterTestInputData(col_ics[i], seed=col_seeds[i], policy=p))
    run!(t)
    @show t.output.nmac==true
end
