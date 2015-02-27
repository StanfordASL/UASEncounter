module EncounterSimulation

using EncounterModel
using EncounterFeatures: AssembledFeatureBlock

export run!, EncounterTest, EncounterTestInputData, EncounterTestOutputData, EncounterPolicy, ConstPolicy, LinearQValuePolicy, make_record, extract_from_record

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
    for i in 1:p.phi.length
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
# function query_policy(p::LinearQFactorExtractionPolicy, state::EncounterState)
#     Qs = 
# end

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
    EncounterTestInputData(initial::EncounterState) = new(-1, INTRUDER, OWNSHIP, SIM, initial, 0, 200, ConstPolicy(HeadingHRL(SIM.legal_D)))
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
    EncounterTest(initial::EncounterState) = new(EncounterTestInputData(initial), EncounterTestOutputData())
end

function gen_test(rng::AbstractRNG)
    t = EncounterTest()
    t.input = EncounterTestInputData()
    t.output = EncounterTestOutputData()

    ix = 1000.0*rand(rng)
    iy = 2000.0*(rand(rng)-0.5)
    ihead = (iy > 0.0 ? -pi*rand(rng) : pi*rand(rng))

    ox = 0.0
    oy = 0.0
    ohead = 0.0

    t.input.id = 0

    t.input.ip = IntruderParams(60.0, 5.0/180.0*pi)
    t.input.op = OwnshipParams(30.0, 45.0/180.0*pi, 1.0)
    t.input.sim = SimParams(1.0, 9.8, [1000.0,0.0], 100.0, 100.0, 0.95)

    t.input.initial = EncounterState([ox,oy,ohead],[ix, iy, ihead],false)
    t.input.random_seed = 0
    t.input.steps = 200

    return t
end

function run!(test::EncounterTest; announce=true)
    if announce
        println("Running test $(test.input.id).")
    end

    function getNextState(s::EncounterState,a::HeadingHRL,rng::AbstractRNG)
        return encounter_dynamics(test.input.sim, test.input.op, test.input.ip, s, a, rng)
    end

    function getReward(s::EncounterState, a::HeadingHRL)
        return reward(test.input.sim, test.input.op, test.input.ip, s, a)
    end

    rng = MersenneTwister(test.input.random_seed)
    r = 0.
    s = test.input.initial
    test.output.states = Array(Any, test.input.steps+1)
    test.output.actions = Array(Any, test.input.steps)
    for i = 1:test.input.steps
        a = query_policy(test.input.policy, s)
        r += getReward(s,a)
        test.output.states[i] = s
        test.output.actions[i] = a
        s = getNextState(s,a,rng)
    end
    a = query_policy(test.input.policy, s)
    test.output.reward = r + getReward(s, a)
    test.output.states[end] = s

    return test
end

end
