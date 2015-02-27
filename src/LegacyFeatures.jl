module LegacyFeatures

using EncounterModel

export f_focused_intruder_grid, f_radial_goal_grid, f_radial_intruder_grid, f_one_over_toca, f_one_over_mindist

function f_radial_intruder_grid(state, 
                                p::Dict{Symbol, Any}=[:num_intruder_dist=>12,
                                                      :max_intruder_dist=>700.0,
                                                      :num_intruder_bearing=>16,
                                                      :num_intruder_heading=>10])
    num_intruder_cells = p[:num_intruder_dist]*p[:num_intruder_bearing]*p[:num_intruder_heading]
    # phi = spzeros(num_intruder_cells, 1)
    phi = zeros(num_intruder_cells)

    if state.end_state
        return phi
    end

    is = state.is
    os = state.os

    d = min(norm(os[1:2]-is[1:2])-SIM.legal_D, p[:max_intruder_dist])

    if d <= 0.0
        return phi
    end

    # bearing to ownship from intruder's perspective
    bearing = atan2(os[2]-is[2], os[1]-is[1]) - is[3]
    while bearing >= 2*pi bearing -= 2*pi end
    while bearing < 0.0 bearing += 2*pi end
    # relative heading of ownship compared to intruder
    heading = os[3] - is[3]
    if heading >= 2*pi heading -= 2*pi end
    if heading < 0.0 heading += 2*pi end
    # FIXME can get rid of this
    @assert heading <= 2*pi && heading >= 0.0 && bearing >= 0.0 && bearing <= 2*pi

    dist_ind = convert(Int64, ceil(min(d/p[:max_intruder_dist],1)*p[:num_intruder_dist]))
    bearing_ind = convert(Int64, ceil(bearing/(2*pi)*p[:num_intruder_bearing]))
    heading_ind = convert(Int64, ceil(heading/(2*pi)*p[:num_intruder_heading]))

    idx = (dist_ind-1)*p[:num_intruder_heading]*p[:num_intruder_bearing] + (bearing_ind-1)*p[:num_intruder_heading] + heading_ind
    # @assert idx <= num_intruder_heading && idx > 0

    # phi[idx, 1] = 1
    phi[idx] = 1

    return phi
end

const num_goal_dist = 10
const max_goal_dist = 500.0
const num_goal_bearing = 15
const num_goal_cells = num_goal_dist*num_goal_bearing

function f_radial_goal_grid(state)
    # phi = spzeros(num_goal_cells, 1)
    phi = zeros(num_goal_cells)

    if state.end_state
        return phi
    end

    os = state.os

    d = min(norm(os[1:2]-SIM.goal_location)-SIM.goal_radius, max_goal_dist)

    if d <= 0.0
        return phi
    end

    heading = atan2(SIM.goal_location[2]-os[2], SIM.goal_location[1]-os[1])
    if heading <= 0.0 heading += 2*pi end # heading now 0 to 2*pi
    bearing = heading-os[3]
    while bearing < 0.0 bearing += 2*pi end
    while bearing >= 2*pi bearing -= 2*pi end

    dist_ind = convert(Int64, ceil(min(d/max_goal_dist,1)*num_goal_dist))
    bearing_ind = convert(Int64, ceil(bearing/(2*pi)*num_goal_bearing))

    idx = (dist_ind-1)*num_goal_bearing + bearing_ind

    # phi[idx, 1] = 1
    phi[idx] = 1

    return phi
end


function f_focused_intruder_grid(state, p::Dict{Symbol, Any}=[:num_intruder_dist=>12,
                                                              :max_intruder_dist=>700.0,
                                                              :num_intruder_bearing=>16,
                                                              :num_intruder_heading=>10])
    num_intruder_cells = p[:num_intruder_dist]*p[:num_intruder_bearing]*p[:num_intruder_heading]
    # phi = spzeros(num_intruder_cells, 1)
    phi = zeros(num_intruder_cells)

    if state.end_state
        return phi
    end

    is = state.is
    os = state.os

    d = min(norm(os[1:2]-is[1:2])-SIM.legal_D, p[:max_intruder_dist])

    if d <= 0.0
        return phi
    end

    # bearing to ownship from intruder's perspective
    bearing = atan2(os[2]-is[2], os[1]-is[1]) - is[3]
    while bearing >= pi bearing -= 2*pi end
    while bearing < -pi bearing += 2*pi end
    # relative heading of ownship compared to intruder
    heading = os[3] - is[3]
    if heading >= 2*pi heading -= 2*pi end
    if heading < 0.0 heading += 2*pi end

    dist_ind = convert(Int64, ceil(min(sqrt(d)/sqrt(p[:max_intruder_dist]),1)*p[:num_intruder_dist]))

    if bearing > pi/2 || bearing < -pi/2
        return phi
    end
    bearing_ind = convert(Int64, ceil((bearing+pi/2)/pi*p[:num_intruder_bearing]))

    heading_ind = convert(Int64, ceil(heading/(2*pi)*p[:num_intruder_heading]))

    idx = (dist_ind-1)*p[:num_intruder_heading]*p[:num_intruder_bearing] + (bearing_ind-1)*p[:num_intruder_heading] + heading_ind
    # @assert idx <= num_intruder_heading && idx > 0

    # phi[idx, 1] = 1
    phi[idx] = 1

    return phi
end

function f_abs_goal_bearing(state)
    if norm(state.os[1:2] - SIM.goal_location) <= SIM.goal_radius
        return [0.0]
    end
    to_goal = atan2(SIM.goal_location[2]-state.os[2], SIM.goal_location[1]-state.os[1])
    bearing = to_goal - state.os[3]
    while bearing > pi bearing -=2.0*pi end
    while bearing < -pi bearing +=2.0*pi end
    return [abs(bearing)]
end

function f_within_goal_dist(state, radius::Float64=100.0)
    return convert(Float64, norm(state.os[1:2] - SIM.goal_location)-SIM.goal_radius <= radius)
end

function f_exp_neg_goal_dist(state::EncounterState, base_dist::Float64=100.0)
    if norm(state.os[1:2] - SIM.goal_location) <= SIM.goal_radius
        return 0.0
    end
    return exp(-(norm(state.os[1:2] - SIM.goal_location)-SIM.goal_radius)/base_dist)
end

function f_intruder_dist(state)
    return dist(state)
end

function f_one_over_dist(state)
    return 1.0/dist(state)
end

function f_one_over_mindist(state)
    return 1.0/mindist(state)
end

function f_exp_neg_dist(state)
    return exp(-dist(state))
end

function f_exp_neg_mindist(state)
    return exp(-mindist(state))
end

function f_one_over_toca(state)
    tau = toca(state)
    if tau < 0.0
        return 0.0
    end
    return 1.0/tau
 end

function f_one_over_mindist_time(state)
    tau = toca(state)
    if tau <= 0.0
        return [0.0]
    end
    return 1.0/(tau*dist(state,tau))
end

function f_mindist_time(state)
    tau = toca(state)
    if tau < 0.0
        tau = 0.0
    end
    return tau*dist(state, tau)
end



end

