import HDF5, JLD
using EncounterModel
using EncounterSimulation

ic_filename = "../data/10k_collisions.ic"
ic_data = JLD.load(ic_filename)

ics = ic_data["ics"]
seeds = ic_data["seeds"]

max_D = 1000.0
min_D = 100.0
target_rr = 0.05
sims_per_iter = 10000

while max_D - min_D > 1.0
    new_D = (max_D + min_D)/2.0
    policy = ConstPolicy(HeadingHRL(new_D))
    # n_nmac = 0
    ts = Array(EncounterTest, sims_per_iter)
    for i in 1:sims_per_iter
        ts[i] = EncounterTest(EncounterTestInputData(ics[i], policy=policy, seed=seeds[i]))
        # run!(t, store_hist=false)
        # if t.output.nmac
        #     n_nmac += 1
        # end
        # print("\rcompleted simulation $i")
    end
    print("\n")
    run!(ts, store_hist=false, parallel=true)
    n_nmac = sum([t.output.nmac for t in ts])
    @show risk_ratio = n_nmac/sims_per_iter
    if risk_ratio > target_rr
        @show min_D = new_D
    else
        @show max_D = new_D
    end
end
