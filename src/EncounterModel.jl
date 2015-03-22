module EncounterModel

import Base.hash

export EncounterAction, EncounterState, PostDecisionState, SimParams, IntruderParams, OwnshipParams, IntruderState, OwnshipState, HeadingHRL, BoundedHeadingHRL, BankControl, RewardModel, DeviationAndTimeReward
export SIM, OWNSHIP, INTRUDER, REWARD
export dist, mindist, toca, encounter_dynamics, reward, ==, hash, post_decision_state, encounter_dynamics
export heading_hrl

typealias OwnshipState AbstractVector{Float64}
# x, y, psi
# typealias OwnshipControl AbstractVector{Float64}
typealias IntruderState AbstractVector{Float64}
# x, y, psi

abstract EncounterAction
abstract HRLAction <: EncounterAction

type EncounterState
    os::OwnshipState
    is::IntruderState
    end_state::Bool
    has_deviated::Bool
end
EncounterState(os::OwnshipState, is::IntruderState, end_state::Bool) = EncounterState(os, is, end_state, false)
typealias PostDecisionState EncounterState

type HeadingHRL <: HRLAction
    D_buffered::Float64
    # hrl::Function
end
==(u::HeadingHRL,v::HeadingHRL) = u.D_buffered==v.D_buffered

type BoundedHeadingHRL <: HRLAction # bounded heading hrl
    D_buffered::Float64
    bound::Float64 # maximum active distance
end

type BankControl <: EncounterAction
    bank::Float64
end
==(u::BankControl,v::BankControl) = u.bank==v.bank

type SimParams
    delta_t::Float64
    g::Float64
    goal_location::Array{Float64,1}
    goal_radius::Float64
    legal_D::Float64
    discount::Float64
end

type IntruderParams
    v::Float64
    heading_std::Float64
end

type OwnshipParams
    v::Float64
    max_phi::Float64
    controller_gain::Float64
    # hrl::Function
end

# const SIM = SimParams(1.0, 9.8, [1000.0, 0.0], 100.0, 100.0, 0.95)
const SIM = SimParams(1.0, 9.8, [1000.0, 0.0], 100.0, 152.4, 1.0) #changed nmac to 500 ft Mar 17
# const OWNSHIP = OwnshipParams(30.0, 45.0/180.0*pi, 1.0)
const OWNSHIP = OwnshipParams(30.0, 45.0/180.0*pi, 3.0)
# const INTRUDER = IntruderParams(60.0, 5.0/180.0*pi)
const INTRUDER = IntruderParams(60.0, 10.0/180.0*pi)

abstract RewardModel

type DeviationAndTimeReward <: RewardModel
    deviation_cost::Float64
    step_cost::Float64
    goal_reward::Float64
    nmac_lambda::Float64
end

# before 3/13
# REWARD = DeviationAndTimeReward(0,1,100,1000)

REWARD = DeviationAndTimeReward(100,1,100,1000)

# utility functions like dist, toca, etc
include("em_util.jl")

# heuristic resolution logic
include("em_hrl.jl")

# dynamics functions
include("em_dynamics.jl")

# reward functions
include("em_reward.jl")

# function hash(a::HeadingHRL)
#     return hash(a.D_buffered)
# end
# 
# function ==(a::HeadingHRL, b::HeadingHRL)
#     return a.D_buffered == b.D_buffered
# end
 
function hash(u::EncounterState)
    if u.end_state
        return hash(u.end_state)
    else
        return hash(u.os, hash(u.is))
    end
end

function ==(u::EncounterState, v::EncounterState)
    if u.end_state && v.end_state
        return true
    elseif u.end_state != v.end_state
        return false
    else
        return u.os==v.os && u.is==v.is && u.has_deviated==v.has_deviated
    end
end

end
