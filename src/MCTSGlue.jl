module MCTSGlue

import Debug

# this won't work
# try
#     import MDP
# catch e
#     println("including mdp within MCTSGlue!")
#     mdp_path = "/home/zach/.julia/mdp/"
#     include("$(mdp_path)src/MDP.jl")
#     include("$(mdp_path)src/mdps/GridWorld.jl")
#     include("$(mdp_path)src/other/auxfuncs.jl")
#     # include("$(mdp_path)src/solvers/MCTSdpw/AG.jl")
#     include("$(mdp_path)src/solvers/MCTSdpw/MCTSdpw.jl")
#     import MDP
# end

import MDP
import MCTSdpw

import Base.hash

using EncounterModel
using EncounterPolicies
# using EncounterSimulation

export MCTSPolicy, query_policy, MDPEncounterState, MDPEncounterAction

type MDPEncounterState <: MDP.State
    s::EncounterState
end
==(u::MDPEncounterState,v::MDPEncounterState) = u.s==v.s
hash(s::MDPEncounterState) = hash(s.s)

type MDPEncounterAction <: MDP.Action
    a::EncounterAction
end
==(u::MDPEncounterAction, v::MDPEncounterAction) = u.a==v.a
hash(a::MDPEncounterAction) = hash(a.a)

# default policy
# for 10 deg use 
function getAction(s::MDPEncounterState, rng::AbstractRNG)
    return MDPEncounterAction(HeadingHRL(1.1*SIM.legal_D)) # I really don't know about this...
end

function getNextState(s::MDPEncounterState, a::MDPEncounterAction, rng::AbstractRNG)
    return MDPEncounterState(encounter_dynamics(SIM, OWNSHIP, INTRUDER, s.s, a.a, rng))
end

type MCTSPolicy <: EncounterPolicy
    d
    ec
    n
    k
    alpha
    kp
    alphap
    rng

    rm::RewardModel
    actions

    s::Dict{MDP.State,MCTSdpw.StateNode} # watch out

    # dpw::MCTSdpw.DPW
end
function MCTSPolicy(d, ec, k, alpha, kp, alphap, n, actions, rm::RewardModel=REWARD, rng=MersenneTwister())
    # p = MCTSdpw.DPWParams(d, ec, n, k, alpha, kp, alphap, rng, getAction, getNextState, getReward, getNextAction)
    # MCTSPolicy(MCTSdpw.DPW(p))
    return MCTSPolicy(d,ec,n,k,alpha,kp,alphap,rng,rm,actions,Dict{MDP.State,MCTSdpw.StateNode}())
end

function query_policy(p::MCTSPolicy, state::EncounterState)
    function getReward(s::MDPEncounterState, a::MDPEncounterAction)
        return reward(p.rm, s.s, a.a)
    end

    # new action
    function getNextAction(s::MDPEncounterState, rng::AbstractRNG)
        return MDPEncounterAction(p.actions[ceil(rand(rng)*length(p.actions))])
    end

    dpw = MCTSdpw.DPW(MCTSdpw.DPWParams(p.d,
                                        p.ec,
                                        p.n,
                                        p.k,
                                        p.alpha,
                                        p.kp,
                                        p.alphap,
                                        p.rng,
                                        getAction,getNextState,getReward,getNextAction))
    dpw.s = p.s

    mdpa = MCTSdpw.selectAction(dpw, MDPEncounterState(state))
    return mdpa.a
end

end
