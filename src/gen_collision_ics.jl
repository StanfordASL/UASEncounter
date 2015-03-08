using EncounterModel
using EncounterSimulation
import HDF5, JLD

N = 10000

collision_fname = "../data/10k_collisions.ic"
mixed_fname = "../data/10k_mixed.ic"

collision_ics = EncounterState[]
collision_seeds = Int64[]

mixed_ics = EncounterState[]
mixed_seeds = Int64[]

policy = ConstPolicy(BankControl(0.0))

gen_rng = MersenneTwister(0)
s=0

first_round = true
first_round_collisions = 0

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
            push!(collision_seeds, s)
        end
        
        print("\rfinished test $i")
        s+=1

        if first_round
            push!(mixed_ics, ic)
            push!(mixed_seeds, s)
        end
    end
    if first_round
        first_round_collisions = length(collision_ics)
        first_round = false
    end
    println("\naccumulated $(length(collision_ics)) collision initial conditions")
end

JLD.save(collision_fname, "ics", collision_ics, "seeds", collision_seeds, "num_collisions", 10000)
JLD.save(mixed_fname, "ics", mixed_ics, "seeds", mixed_seeds, "num_collisions", first_round_collisions)
