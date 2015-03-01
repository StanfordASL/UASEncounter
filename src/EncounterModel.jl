module EncounterModel

import Base.hash

export EncounterAction, EncounterState, PostDecisionState, SimParams, IntruderParams, OwnshipParams, IntruderState, OwnshipState, HeadingHRL, BankControl
export SIM, OWNSHIP, INTRUDER
export dist, mindist, toca, encounter_dynamics, reward, ==, hash

typealias OwnshipState AbstractVector{Float64}
# x, y, psi
# typealias OwnshipControl AbstractVector{Float64}
typealias IntruderState AbstractVector{Float64}
# x, y, psi

abstract EncounterAction

type EncounterState
    os::OwnshipState
    is::IntruderState
    end_state::Bool
end
typealias PostDecisionState EncounterState

type HeadingHRL <: EncounterAction
    D_buffered::Float64
    # hrl::Function
end

type BankControl <: EncounterAction
    bank::Float64
end

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

const SIM = SimParams(1.0, 9.8, [1000.0, 0.0], 100.0, 100.0, 0.95)
# const OWNSHIP = OwnshipParams(30.0, 45.0/180.0*pi, 1.0)
const OWNSHIP = OwnshipParams(30.0, 45.0/180.0*pi, 3.0)
# const INTRUDER = IntruderParams(60.0, 5.0/180.0*pi)
const INTRUDER = IntruderParams(60.0, 15.0/180.0*pi)

# utility functions like dist, toca, etc
include("em_util.jl")

# heuristic resolution logic
include("em_hrl.jl")

# dynamics functions
include("em_dynamics.jl")

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
        return u.os==v.os && u.is == v.is
    end
end

end
