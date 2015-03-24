using EncounterModel
using EncounterSimulation
import HDF5, JLD
import ArgParse

s = ArgParse.ArgParseSettings()

ArgParse.@add_arg_table s begin
    "--ndeg"
        help = "intruder noise standard dev in degrees"
        arg_type = Float64
        default = 10.0
end

args = ArgParse.parse_args(s)

ndeg_arg = args["ndeg"]

N = 10000

# collision_fname = "../data/10k_collisions.ic"
# mixed_fname = "../data/10k_mixed.ic"
# collision_fname = "../data/box_10k_collisions.ic"
# mixed_fname = "../data/box_10k_mixed.ic"
collision_fname = "../data/box_10k_collisions_$(int(ndeg_arg)).ic"
mixed_fname = "../data/box_10k_mixed_$(int(ndeg_arg)).ic"

INTRUDER.heading_std=args["ndeg"]*pi/180.0

collision_ics = EncounterState[]
collision_seeds = Int64[]

mixed_ics = EncounterState[]
mixed_seeds = Int64[]

policy = ConstPolicy(BankControl(0.0))

gen_rng = MersenneTwister(0)
s=0

first_round = true
first_round_collisions = 0

qxys = [1=>[:xmin=>1300,:xmax=>1700,:ymin=>-800,:ymax=>800],
        2=>[:xmin=>-300,:xmax=>1300,:ymin=>800,:ymax=>1200],
        3=>[:xmin=>-700,:xmax=>-300,:ymin=>-800,:ymax=>800],
        4=>[:xmin=>-300,:xmax=>1300,:ymin=>-1200,:ymax=>-800],
       ]

           
function gen_box_state(rng::AbstractRNG)
    # first choose quadrant
    qscores = zeros(4)
    rand!(rng, qscores)
    q = indmax(qscores)
    x = rand(rng)*(qxys[q][:xmax]-qxys[q][:xmin])+qxys[q][:xmin]
    y = rand(rng)*(qxys[q][:ymax]-qxys[q][:ymin])+qxys[q][:ymin]
    heading = pi*rand(rng) + pi/2*q
    
    return EncounterState([0,0,0],[x,y,heading],false,false)
end

while length(collision_ics) < N
    nround = N-length(collision_ics)
    println("running $nround tests...")
    refs = Array(Any, nround)
    ts = Array(EncounterTest,nround)
    ics = Array(EncounterState,nround)
    seeds = rand(Uint32, nround)
    for i = 1:nround
        # ic = gen_init_state(gen_rng) 
        ics[i] = gen_box_state(gen_rng) 
        ts[i] = EncounterTest(EncounterTestInputData(ics[i], policy=policy, seed=seeds[i]))
    end

    run!(ts, store_hist=false, parallel=true)

    for i = 1:nround
        t = ts[i]
        ic = ics[i]
        s = seeds[i]

        if t.output.nmac
            push!(collision_ics, ic)
            push!(collision_seeds, s)
        end
        
        print("\rfinished test $i")

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
