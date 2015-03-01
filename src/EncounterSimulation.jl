module EncounterSimulation

using EncounterModel
using EncounterFeatures: AssembledFeatureBlock

export run!, EncounterTest, EncounterTestInputData, EncounterTestOutputData, EncounterPolicy, ConstPolicy, LinearQValuePolicy, make_record, extract_from_record, gen_init_state

abstract EncounterPolicy
function make_record(policy::EncounterPolicy)
    return policy # many policies can probably be stored without needing any manipulation
end
function extract_from_record(record::EncounterPolicy)
    return record
end

type LinearQValuePolicy <: EncounterPolicy
    phi::AssembledFeatureBlock
    actions::Vector{EncounterAction}
    lambdas::Vector{Vector{Float64}}
end
function query_policy(p::LinearQValuePolicy, state::EncounterState)
    qs=Array(Float64, length(p.actions))
    for i in 1:length(qs)
        qs[i] = sum(p.phi.features(state)'*p.lambdas[i])
    end
    return p.actions[indmax(qs)]
end

type LinearQValuePolicyRecord
    phi_desc::Vector{String}
    actions::Vector{EncounterAction}
    lambdas::Vector{Vector{Float64}}
end
function make_record(p::LinearQValuePolicy)
    return LinearQValuePolicyRecord(p.phi.description, p.actions, p.lambdas)
end
function extract_from_record(record::LinearQValuePolicyRecord)
    return LinearQValuePolicy(AssembledFeatureBlock(record.phi_desc), record.actions, record.lambdas)
end

type ConstPolicy <: EncounterPolicy
    action::EncounterAction
end
function query_policy(p::ConstPolicy, state::EncounterState)
    return p.action
end

type EncounterTestInputData
    id

    ip::IntruderParams
    op::OwnshipParams
    sim::SimParams
    initial::EncounterState
    random_seed::Int
    steps::Int

    policy::EncounterPolicy

    EncounterTestInputData() = new()
    EncounterTestInputData(initial::EncounterState; policy::EncounterPolicy=ConstPolicy(HeadingHRL(SIM.legal_D)), seed::Int=0,id=nothing) = new(id,INTRUDER, OWNSHIP, SIM, initial, seed, 200, policy)
end
type EncounterTestInputRecord # can be saved to disk
    id

    ip::IntruderParams
    op::OwnshipParams
    sim::SimParams
    initial::EncounterState
    random_seed::Int
    steps::Int

    policy_record
end
function make_record(tid::EncounterTestInputData)
    return EncounterTestInputRecord(tid.id, tid.ip, tid.op, tid.sim, tid.initial, tid.random_seed, tid.steps, make_record(tid.policy))
end
function extract_from_record(r::EncounterTestInputRecord)
    return EncounterTestInputData(r.id, r.ip, r.op, r.sim, r.initial, r.random_seed, r.steps, extract_from_record(r.policy_record))
end

type EncounterTestOutputData
    states::Array{Any, 1}
    actions::Array{Any, 1}
    reward::Float64

    EncounterTestOutputData() = new() 
end

type EncounterTest
    input::EncounterTestInputData
    output::EncounterTestOutputData

    EncounterTest() = new()
    EncounterTest(input::EncounterTestInputData) = new(input, EncounterTestOutputData())
    EncounterTest(initial::EncounterState) = new(EncounterTestInputData(initial), EncounterTestOutputData())
end
type EncounterTestRecord
    input_record::EncounterTestInputRecord
    output::EncounterTestOutputData
end
function make_record(t::EncounterTest)
    return EncounterTestRecord(make_record(t.input), t.output)
end
function extract_from_record(r::EncounterTestRecord)
    return EncounterTest(extract_from_record(r.input_record), r.output)
end

function gen_init_state(rng::AbstractRNG)
    ix = 1000.0*rand(rng)
    iy = 2000.0*(rand(rng)-0.5)
    ihead = (iy > 0.0 ? -pi*rand(rng) : pi*rand(rng))

    ox = 0.0
    oy = 0.0
    ohead = 0.0

    return EncounterState([ox,oy,ohead],[ix, iy, ihead],false)
end


# function gen_test(rng::AbstractRNG, policy::EncounterPolicy)
#     t = EncounterTest()
#     t.input = EncounterTestInputData()
#     t.output = EncounterTestOutputData()
# 
#     ix = 1000.0*rand(rng)
#     iy = 2000.0*(rand(rng)-0.5)
#     ihead = (iy > 0.0 ? -pi*rand(rng) : pi*rand(rng))
# 
#     ox = 0.0
#     oy = 0.0
#     ohead = 0.0
# 
#     t.input.id = 0
# 
#     t.input.ip = INTRUDER
#     t.input.op = OWNSHIP
#     t.input.sim = SIM
# 
#     t.input.initial = EncounterState([ox,oy,ohead],[ix, iy, ihead],false)
#     t.input.random_seed = 0
#     t.input.steps = 200
# 
#     t.input.policy = EncounterPolicy
# 
#     return t
# end

function run!(test::EncounterTest; announce=true, store_hist=true)
    if announce
        println("Running test $(test.input.id).")
    end

    function getNextState(s::EncounterState,a::EncounterAction,rng::AbstractRNG)
        return encounter_dynamics(test.input.sim, test.input.op, test.input.ip, s, a, rng)
    end

    function getReward(s::EncounterState, a::EncounterAction)
        return reward(test.input.sim, test.input.op, test.input.ip, s, a)
    end

    rng = MersenneTwister(test.input.random_seed)
    r = 0.
    s = test.input.initial
    if store_hist
        test.output.states = Array(Any, test.input.steps+1)
        test.output.actions = Array(Any, test.input.steps)
    else
        test.output.states = {}
        test.output.actions = {}
    end

    for i = 1:test.input.steps
        a = query_policy(test.input.policy, s)
        r += getReward(s,a)
        if store_hist
            test.output.states[i] = s
            test.output.actions[i] = a
        end
        s = getNextState(s,a,rng)
    end
    a = query_policy(test.input.policy, s)
    test.output.reward = r + getReward(s, a)
    if store_hist
        test.output.states[end] = s
    end

    return test
end

function run!(tests::Vector{EncounterTest}; store_hist=true, parallel=false, batch_size=100)
    if parallel
        num_batches = int(ceil(length(tests)/batch_size))

        println("spawning $(length(tests)) simulations...")
        refs = Array(Any, num_batches)
        for b in 1:num_batches
            test_range = (b-1)*batch_size+1:min(length(tests),b*batch_size)
            batch = tests[test_range]
            refs[b] = @spawn run!(batch, parallel=false, store_hist=store_hist)
        end
        for b in 1:num_batches 
            test_range = (b-1)*batch_size+1:min(length(tests),b*batch_size)
            print("\rwaiting for test batch $b of $num_batches...")
            tests[test_range] = fetch(refs[b])
        end
        println("\rdone with simulations")
    else
        for i in 1:length(tests)
            run!(tests[i]; announce=false, store_hist=store_hist)
        end
    end
    return tests
end

end
