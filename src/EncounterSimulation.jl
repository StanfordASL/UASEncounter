module EncounterSimulation

# import Debug

using EncounterModel
using EncounterFeatures
using EncounterPolicies
using MCTSGlue
export run!, EncounterTest, EncounterTestInputData, EncounterTestOutputData, test_policy
# export run!, EncounterTest, EncounterTestInputData, EncounterTestOutputData, EncounterPolicy, ConstPolicy, LinearQValuePolicy, make_record, extract_from_record, gen_init_state, query_policy_ind, test_policy

type EncounterTestInputData
    id

    ip::IntruderParams
    op::OwnshipParams
    sim::SimParams
    initial::EncounterState
    random_seed
    steps::Int

    rm::RewardModel
    policy::EncounterPolicy

    EncounterTestInputData() = new(nothing,
                                   INTRUDER,OWNSHIP,SIM,
                                   EncounterState([0,0,0],[0,0,0],false,false),
                                   0, 200, REWARD,ConstPolicy(BankControl(0.0)))
    EncounterTestInputData(initial::EncounterState;
                           policy::EncounterPolicy=ConstPolicy(BankControl(0.0)),
                           seed=0,
                           rm::RewardModel=REWARD,
                           id=nothing) = new(id,INTRUDER, OWNSHIP, SIM, initial, seed, 200, rm, policy)
end

type EncounterTestOutputData
    states::Array{Any, 1}
    actions::Array{Any, 1}
    reward::Float64
    nmac::Bool
    deviated::Bool
    steps_before_end::Int64

    EncounterTestOutputData() = new({},{},0.0,false,false, 0) 
end

type EncounterTest
    input::EncounterTestInputData
    output::EncounterTestOutputData

    EncounterTest() = new()
    EncounterTest(input::EncounterTestInputData) = new(input, EncounterTestOutputData())
    EncounterTest(initial::EncounterState) = new(EncounterTestInputData(initial), EncounterTestOutputData())
end

function gen_init_state(rng::AbstractRNG)
    # ix = 1000.0*rand(rng)
    # iy = 2000.0*(rand(rng)-0.5)
    # ihead = (iy > 0.0 ? -pi*rand(rng) : pi*rand(rng))

    min_dist = 800.0
    max_dist = 1200.0
    dist = (max_dist-min_dist)*rand(rng)+min_dist
    bearing = 2*pi*rand(rng)
    ix = dist*cos(bearing)
    iy = dist*sin(bearing)

    ihead = atan2(-iy, 100.0-ix) + 20.0*pi/180.0*randn(rng)
    # ihead = (iy > 0.0 ? -pi*rand(rng) : pi*rand(rng))

    ox = 0.0
    oy = 0.0
    ohead = 0.0

    return EncounterState([ox,oy,ohead],[ix, iy, ihead],false,false)
end

function run!(test::EncounterTest; announce=false, store_hist=true)
    if announce
        println("Running test $(test.input.id).")
    end

    function getNextState(s::EncounterState,a::EncounterAction,rng::AbstractRNG)
        return encounter_dynamics(test.input.sim, test.input.op, test.input.ip, s, a, rng)
    end

    function getReward(s::EncounterState, a::EncounterAction)
        return reward(test.input.sim, test.input.op, test.input.ip, test.input.rm, s, a)
    end

    rng = MersenneTwister(test.input.random_seed)
    r = 0.
    s = test.input.initial

    test.output.nmac = false
    if store_hist
        test.output.states = Array(Any, test.input.steps+1)
        test.output.actions = Array(Any, test.input.steps)
    else
        test.output.states = {}
        test.output.actions = {}
    end

    for i = 1:test.input.steps
        if !s.end_state
            test.output.steps_before_end += 1
        end

        a = query_policy(test.input.policy, s)
        r += getReward(s,a)
        if store_hist
            test.output.states[i] = s
            test.output.actions[i] = a
        end
        s = getNextState(s,a,rng)

        if !s.end_state && dist(s) <= test.input.sim.legal_D
            test.output.nmac = true
        end
        if s.has_deviated
            test.output.deviated = true
        end
    end
    a = query_policy(test.input.policy, s)
    test.output.reward = r + getReward(s, a)
    if store_hist
        test.output.states[end] = s
    end

    return test
end

function run!(tests::Vector{EncounterTest}; store_hist=false, parallel=false, batch_size=100)
    if parallel
        # num_batches = int(ceil(length(tests)/batch_size))

        println("spawning $(length(tests)) simulations...")
        # refs = Array(Any, num_batches)
        # for b in 1:num_batches
        #     test_range = (b-1)*batch_size+1:min(length(tests),b*batch_size)
        #     batch = tests[test_range]
        #     refs[b] = @spawn run!(batch, parallel=false, store_hist=store_hist)
        # end
        # for b in 1:num_batches 
        #     test_range = (b-1)*batch_size+1:min(length(tests),b*batch_size)
        #     print("\rwaiting for test batch $b of $num_batches...")
        #     tests[test_range] = fetch(refs[b])
        # end
        
        results = pmap(run!, tests, err_stop=true)
        tests[:] = results

        println("\rdone with simulations")
    else
        for i in 1:length(tests)
            run!(tests[i]; announce=false, store_hist=store_hist)
        end
    end
    return tests
end

function test_policy(policy::EncounterPolicy, ics, seeds; store_hist=false, parallel=true)
    ts = Array(EncounterTest, length(ics))
    for i in 1:length(ics)
        ts[i] = EncounterTest(EncounterTestInputData(ics[i], policy=policy, seed=seeds[i]))
    end
    run!(ts, store_hist=store_hist, parallel=parallel)
    return ts
end

end
