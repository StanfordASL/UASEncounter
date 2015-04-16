module EncounterPolicies

using EncounterModel
using EncounterFeatures
export EncounterPolicy, LinearQValuePolicy, LinearPostDecisionPolicy, ConstPolicy
export query_policy, query_policy_ind

abstract EncounterPolicy

type LinearQValuePolicy <: EncounterPolicy
    phi::FeatureBlock
    actions::Vector{EncounterAction}
    thetas::Vector{Vector{Float64}}
end
function query_policy_ind(p::LinearQValuePolicy, state::EncounterState)
    qs=Array(Float64, length(p.actions))
    for i in 1:length(qs)
        qs[i] = sum(evaluate(p.phi,state)'*p.thetas[i])
    end
    return indmax(qs)
end
function query_policy(p::LinearQValuePolicy, state::EncounterState)
    return p.actions[query_policy_ind(p,state)]
end

type LinearPostDecisionPolicy <: EncounterPolicy
    phi::FeatureBlock
    actions::Vector{EncounterAction}
    theta::Vector{Float64}
    rm::RewardModel
end
function query_policy_ind(p::LinearPostDecisionPolicy, state::EncounterState)
    pdvals=Array(Float64, length(p.actions))
    for i in 1:length(pdvals)
        pd = post_decision_state(state, p.actions[i])
        pdvals[i] = reward(p.rm, state, p.actions[i]) + sum(evaluate(p.phi,pd)'*p.theta)
    end
    return indmax(pdvals)
end
function query_policy(p::LinearPostDecisionPolicy, state::EncounterState)
    return p.actions[query_policy_ind(p,state)]
end

type ConstPolicy <: EncounterPolicy
    action::EncounterAction
end
function query_policy(p::ConstPolicy, state::EncounterState)
    return p.action
end


end
