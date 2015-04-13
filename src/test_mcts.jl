using EncounterModel
import MDP #WHY THE HECK DOES THIS HELP????
using MCTSGlue
import Base.Test

import MCTSdpw

d = 70
ec = 100
k = 5
alpha = 0.5
kp = 5
alphap = 0.5
n = 50

lD = SIM.legal_D
actions = EncounterAction[HeadingHRL(D) for D in [0.0, lD, 1.5*lD, 2.0*lD, 3.0*lD, 4.0*lD]]

dynrng = MersenneTwister(0)
rng1 = MersenneTwister(0)
rng2 = MersenneTwister(0)

policy = MCTSPolicy(d, ec, k, alpha, kp, alphap, n, actions, REWARD, rng1)

function getReward(s::MDPEncounterState, a::MDPEncounterAction)
    return reward(policy.rm, s.s, a.a)
end

# new action
function getNextAction(s::MDPEncounterState, rng::AbstractRNG)
    return MDPEncounterAction(policy.actions[ceil(rand(rng)*length(policy.actions))])
end

function getAction(s::MDPEncounterState, rng::AbstractRNG)
    return MDPEncounterAction(HeadingHRL(1.1*SIM.legal_D)) # I really don't know about this...
end

function getNextState(s::MDPEncounterState, a::MDPEncounterAction, rng::AbstractRNG)
    return MDPEncounterState(encounter_dynamics(SIM, OWNSHIP, INTRUDER, s.s, a.a, rng))
end

dpw = MCTSdpw.DPW(MCTSdpw.DPWParams(d,ec,n,k,alpha,kp,alphap,rng2,getAction,getNextState,getReward,getNextAction))


state = EncounterState([0,0,0], [600,-250,135*pi/180],false,false)

for t in 0:10
    a2 = MCTSdpw.selectAction(dpw, MDPEncounterState(state)).a
    a1 = query_policy(policy, state)
    Test.@test a1==a2
    state = encounter_dynamics(SIM,OWNSHIP,INTRUDER, state, a1, dynrng)
end
