
function reward(sim::SimParams, op::OwnshipParams, ip::IntruderParams, rm::RewardModel, state::EncounterState, action::EncounterAction)
    if state.end_state
        return 0.0
    end

    r = -rm.step_cost
    if norm(state.os[1:2]-sim.goal_location) <= sim.goal_radius
        r+=rm.goal_reward
    end
    if dist(state) <= sim.legal_D
        r-=rm.nmac_lambda
    end

    if !state.has_deviated
        if ownship_control(op, ip, state, action).bank > 1e-5
            r-=rm.deviation_cost
        end
    end
    return r
end
reward(rm::RewardModel, state::EncounterState, action::EncounterAction) = reward(SIM, OWNSHIP, INTRUDER, rm, state, action)

# type LegacyRewardModel
# 
# end
# 
# function reward(sim::SimParams, op::OwnshipParams, ip::IntruderParams, state::EncounterState, action::EncounterAction)
#     if state.end_state
#         return 0.0
#     end
# 
#     # r = 0.0
#     r = -10.0
#     if norm(state.os[1:2]-sim.goal_location) <= sim.goal_radius
#         r+=1000
#     end
#     if dist(state) <= sim.legal_D
#         r-=10000
#     end
# 
#     # desired_heading = hrl(state, action, op, ip)
#     # ctrl = ownship_control(op, state.os, desired_heading)
# 
#     # r -= 5.0*abs(ctrl[1])
#     return r
# end
# reward(state::EncounterState) = reward(SIM, OWNSHIP, INTRUDER, state, BankControl(0.0))


