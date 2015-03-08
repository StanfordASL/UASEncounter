using EncounterModel
using EncounterSimulation
import HDF5, JLD

N = 10000

filename = "../data/10k_collisions.ic"

collision_ics = {}
seeds = Int64[]

policy = ConstPolicy(BankControl(0.0))

gen_rng = MersenneTwister(0)
s=0

while length(collision_ics) < N
    nround = N-length(collision_ics)
    println("running $nround tests...")
    refs = Array(Any, nround)
    for i = 1:nround
        ic = gen_init_state(gen_rng) 
        t = EncounterTest(EncounterTestInputData(ic, policy=policy, seed=s))
        run!(t, store_hist=false)
        if t.output.nmac
            push!(collision_ics, ic)
            push!(seeds, s)
        end
        print("\rfinished test $i")
        s+=1
    end
    println("\naccumulated $(length(collision_ics)) collision initial conditions")
end

JLD.save(filename, "ics", collision_ics, "seeds", seeds)
