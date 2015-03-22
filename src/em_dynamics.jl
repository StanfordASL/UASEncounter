# included in EncounterModel

function encounter_dynamics(sim::SimParams, op::OwnshipParams, ip::IntruderParams, state::EncounterState, action::EncounterAction, rng::AbstractRNG)
    pd = post_decision_state(sim, op, ip, state, action)
    return next_state_from_pd(sim, ip, pd, rng)
end
# encounter_dynamics(hrl::Function, state::EncounterState) = encounter_dynamics(SIM, OWNSHIP, INTRUDER, hrl, state)

function next_state_from_pd(sim::SimParams, ip::IntruderParams, pd::PostDecisionState, rng::AbstractRNG)
    next_state = EncounterState(copy(pd.os), Array(Float64, 3), pd.end_state, pd.has_deviated)
    intruder_dynamics!(next_state.is, sim, ip, pd.is, randn(rng))
    return next_state
end
next_state_from_pd(pd::PostDecisionState, rng::AbstractRNG) = next_state_from_pd(SIM, INTRUDER, pd, rng)

# function post_decision_state(sim::SimParams, op::OwnshipParams, ip::IntruderParams, state::EncounterState, action::HeadingHRL)
#     pd = PostDecisionState(Array(Float64,3), copy(state.is), false)
# 
#     if state.end_state || norm(state.os[1:2]-sim.goal_location) <= sim.goal_radius || norm(state.os[1:2]-state.is[1:2]) <= sim.legal_D
#         pd.end_state = true
#         return pd
#     end
# 
#     desired_heading = heading_hrl(state, action.D_buffered, op, ip)
#     ctrl = ownship_control(op, state.os, desired_heading)
#     ownship_dynamics!(pd.os, sim, op, state.os, ctrl)
#     return pd
# end
post_decision_state(state::EncounterState, action::EncounterAction) = post_decision_state(SIM, OWNSHIP, INTRUDER, state, action)

function post_decision_state(sim::SimParams, op::OwnshipParams, ip::IntruderParams, state::EncounterState, action::EncounterAction)
    pd = PostDecisionState(Array(Float64,3), copy(state.is), false, state.has_deviated)

    if state.end_state || norm(state.os[1:2]-sim.goal_location) <= sim.goal_radius || norm(state.os[1:2]-state.is[1:2]) <= sim.legal_D
        pd.end_state = true
        return pd
    end

    ctrl = ownship_control(op, ip, state, action)
    if abs(ctrl.bank) > 1e-5
        pd.has_deviated = true
    end
    ownship_dynamics!(pd.os, sim, op, state.os, ctrl)
    return pd
end

function ownship_control(op::OwnshipParams, ip::IntruderParams, state::EncounterState, action::BankControl)
    return action
end

function ownship_control(op::OwnshipParams, ip::IntruderParams, state::EncounterState, action::HRLAction)
#     @show state
#     @show desired_heading
    desired_heading = heading_hrl(state, action, op, ip)
    diff = desired_heading-state.os[3]
    while diff > pi; diff-=2*pi; end
    while diff < -pi; diff+=2*pi; end
    ctrl = min(op.max_phi, max(-op.max_phi, op.controller_gain*(diff)))
    return BankControl(ctrl)
end
ownship_control(state::OwnshipState, desired_heading::Float64) = ownship_control(OWNSHIP, state, desired_heading)

function ownship_dynamics!(nextstate::OwnshipState, sim::SimParams, op::OwnshipParams, state::OwnshipState, ctrl::BankControl)
    psi = state[3]
    v = op.v
    phi = ctrl.bank
    psidot = sim.g*tan(phi)/v
    nextpsi = psi + psidot*sim.delta_t
    eps = 1e-5
    if abs(psidot) <= eps
        nextstate[1] = state[1] + v*cos(psi)*sim.delta_t
        nextstate[2] = state[2] + v*sin(psi)*sim.delta_t
    else
        nextstate[1] = state[1] + v*(sin(nextpsi) - sin(psi))/psidot
        nextstate[2] = state[2] - v*(cos(nextpsi) - cos(psi))/psidot
    end
    nextstate[3] = nextpsi
    return nothing
end

function ownship_dynamics(sim::SimParams, op::OwnshipParams, state::OwnshipState, ctrl::BankControl)
    next_state = Array(Float64,3)
    ownship_dynamics!(next_state, sim, op, state, ctrl)
    return next_state
end
ownship_dynamics(state::OwnshipState, ctrl::BankControl) = ownship_dynamics(SIM, OWNSHIP, state, ctrl)

function intruder_dynamics!(nextstate::IntruderState, sim::SimParams, ip::IntruderParams, state::IntruderState, unscaled_heading_noise::Float64)
    psidot = ip.heading_std*unscaled_heading_noise
    psi = state[3]
    nextpsi = psi + psidot*sim.delta_t
    eps = 1e-5
    if abs(psidot) <= eps
        nextstate[1] = state[1] + ip.v*cos(psi)*sim.delta_t
        nextstate[2] = state[2] + ip.v*sin(psi)*sim.delta_t
    else
        nextstate[1] = state[1] + ip.v*(sin(nextpsi) - sin(psi))/psidot
        nextstate[2] = state[2] - ip.v*(cos(nextpsi) - cos(psi))/psidot
    end
    nextstate[3] = nextpsi
    return nothing
end


