module EncounterFeatures

using EncounterModel: EncounterState, SIM, dist, mindist, toca
using GridInterpolations

using LegacyFeatures
import LegacyFeatures.f_radial_goal_grid

# import Base.Test.@test
import Base.length

export ParameterizedFeatureBlock, ParameterizedFeatureFunction
export evaluate
export FeatureBlock
export GOAL_GRID, INTRUDER_GRID, FEATURES
export f_in_goal, f_mindist_time, f_one_over_mindist_time, f_exp_neg_mindist, f_exp_neg_dist, f_one_over_mindist, f_one_over_dist, f_intruder_dist, f_one, f_goal_dist, f_abs_goal_bearing, f_radial_intruder_grid, f_radial_goal_grid, f_exp_neg_goal_dist, f_within_goal_dist, f_conflict, f_focused_intruder_grid, f_half_intruder_bin_grid, f_has_deviated

type FeatureBlock
    members::Vector{Any}
    uses_mem::Bool
end
function FeatureBlock{T}(members::Vector{T})
    mymem = {}
    for m in members
        if isa(m,Symbol)
            push!(mymem,FeatureFunction(m)) 
        else
            push!(mymem,m)
        end
    end
    return FeatureBlock(mymem, any([uses_mem(m) for m in mymem]))
end

length(b::FeatureBlock) = sum([length(m) for m in b.members])
uses_mem(b::FeatureBlock) = b.uses_mem


# fun returns a single element vector, there are many param values
type ParameterizedFeatureBlock
    fun::Symbol
    params::Vector{Any}
end

length(b::ParameterizedFeatureBlock) = length(b.params)
uses_mem(b::ParameterizedFeatureBlock) = false

# fun may return a multi-element vector, there is only one param value
type ParameterizedFeatureFunction
    fun::Symbol
    param::Any

    uses_mem::Bool
    length::Int64
end
ParameterizedFeatureFunction(fun::Symbol, param::Any) = ParameterizedFeatureFunction(fun, param, false)
function ParameterizedFeatureFunction(fun::Symbol, param::Any, memory::Bool)
    ret = eval(fun)(gen_test_state(), param)
    return ParameterizedFeatureFunction(fun, param, memory, length(ret))
end

length(b::ParameterizedFeatureFunction) = b.length
uses_mem(b::ParameterizedFeatureFunction) = b.uses_mem

type FeatureFunction
    fun::Symbol
end
uses_mem(f::FeatureFunction) = false
length(f::FeatureFunction) = 1


function evaluate(block::ParameterizedFeatureBlock, state::EncounterState)
    b = Array(Float64, length(block.params))
    fun = eval(block.fun)
    for i in 1:length(block.params)
        b[i:i] = fun(state, block.params[i])
    end
    return b
end

function evaluate(f::ParameterizedFeatureFunction, state::EncounterState; memory::AbstractVector{Float64}=Float64[]) 
    return eval(f.fun)(state, f.param, memory=memory)
end

function evaluate(f::FeatureFunction, state::EncounterState)
    return eval(f.fun)(state)
end

function evaluate(block::FeatureBlock, state::EncounterState; memory::AbstractVector{Float64}=Float64[])
    b = memory
    if length(memory)==0
        b = zeros(Float64, length(block))
        # b = Array(Float64, sum([a.length for a in blocks]))
    # elseif length(memory)==0
    #     b = spzeros(sum([a.length for a in blocks]),1)
    end
    if end_state
        b[:] = 0
        return b
    end
    i = 1
    for a in block.members
        if uses_mem(a)
            evaluate(a, state, memory=sub(b, i:i-1+length(a)))
        else
            b[i:i-1+length(a)] = evaluate(a, state)
        end
        i+=length(a)
    end
    return b
end

function gen_test_state()
    return EncounterState([0.,0.,0.], [0.,0.,0.],true)
end

function f_focused_intruder_grid(state, grid::AbstractGrid; memory::AbstractVector{Float64}=Float64[])
    # phi = spzeros(length(grid), 1)
    phi = memory
    if length(memory)==0
        phi = zeros(length(grid))
    end
        
    if state.end_state
        return phi
    end
    is = state.is
    os = state.os

    d = min(norm(os[1:2]-is[1:2])-SIM.legal_D, maximum(grid.cutPoints[1]))

    # bearing to ownship from intruder's perspective
    bearing = atan2(os[2]-is[2], os[1]-is[1]) - is[3]
    while bearing >= pi bearing -= 2*pi end
    while bearing < -pi bearing += 2*pi end
    # relative heading of ownship compared to intruder
    heading = os[3] - is[3]
    while heading >= 2*pi heading -= 2*pi end
    while heading < 0.0 heading += 2*pi end

    if bearing > pi/2 || bearing < -pi/2 || d <= 0.0
        return phi
    end

    inds, weights = interpolants(grid, [d, bearing, heading])
    # if maximum(weights) < 0.95
    #     @show bearing
    #     @show heading
    #     @show d
    #     @show ind2x(grid, inds[indmax(weights)])
    # end
    # @assert maximum(weights) > 0.95

    phi[inds] = weights
    # for i in 1:length(inds)
    #     phi[inds[i]] = weights[i]
    # end
    return phi
end

function f_half_intruder_bin_grid(state, p::Dict{Symbol, Any}=[:num_intruder_dist=>12,
                                                              :max_intruder_dist=>700.0,
                                                              :num_intruder_bearing=>16,
                                                              :num_intruder_heading=>10];
                                                              memory::AbstractVector{Float64}=Float64[])
    num_intruder_cells = p[:num_intruder_dist]*p[:num_intruder_bearing]*p[:num_intruder_heading]
    # phi = spzeros(num_intruder_cells, 1)
    phi = memory
    if length(memory) == 0
        phi = zeros(num_intruder_cells)
    end

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
    while heading >= 2*pi heading -= 2*pi end
    while heading < 0.0 heading += 2*pi end

    dist_ind = convert(Int64, ceil(min(d/p[:max_intruder_dist],1)*p[:num_intruder_dist]))
    # dist_ind = convert(Int64, ceil(min(sqrt(d)/sqrt(p[:max_intruder_dist]),1)*p[:num_intruder_dist]))

    if bearing > pi/2 || bearing < -pi/2
        return phi
    end
    bearing_ind = convert(Int64, ceil((bearing+pi/2)/pi*p[:num_intruder_bearing]))

    heading_ind = convert(Int64, ceil(heading/(2*pi)*p[:num_intruder_heading]))

    idx = (dist_ind-1)*p[:num_intruder_heading]*p[:num_intruder_bearing] + (bearing_ind-1)*p[:num_intruder_heading] + heading_ind
    # @assert idx <= num_intruder_heading && idx > 0

    if idx < 1 || idx > length(phi)
        @show bearing_ind, heading_ind, dist_ind
        @show num_intruder_cells
        @show length(phi)
        @show idx
    end

    # phi[idx, 1] = 1
    phi[idx] = 1

    return phi
end

function f_radial_goal_grid(state, grid::AbstractGrid; memory::AbstractVector{Float64}=Float64[])
    # phi = spzeros(length(grid),1)
    phi = memory
    if length(memory)==0
        phi = zeros(length(grid))
    end

    if state.end_state
        return phi
    end
    os = state.os

    d = min(norm(os[1:2]-SIM.goal_location)-SIM.goal_radius, maximum(grid.cutPoints[1]))

    if d <= 0.0
        return phi
    end

    heading = atan2(SIM.goal_location[2]-os[2], SIM.goal_location[1]-os[1])
    if heading <= 0.0 heading += 2*pi end # heading now 0 to 2*pi
    bearing = heading-os[3]
    while bearing < -pi bearing += 2*pi end
    while bearing > pi bearing -= 2*pi end
    # while bearing < 0.0 bearing += 2*pi end
    # while bearing >= 2*pi bearing -= 2*pi end

    inds, weights = interpolants(grid, [d, bearing])
    # if maximum(weights) < 0.95
    #     @show state
    #     @show d,heading,bearing
    #     @show ind2x(grid, inds[indmax(weights)])
    # end

    for i in 1:length(inds)
        phi[inds[i]] = weights[i]
    end
    return phi
end

function f_has_deviated(state::EncounterState)
    return convert(Float64, state.has_deviated)
end

function f_conflict(state::EncounterState)
    if state.end_state
        return 0.0
    end

    return convert(Float64, norm(state.os[1:2]-state.is[1:2])<=SIM.legal_D)
end

function f_in_goal(state::EncounterState)
    # dist = sqrt(sum((state.os[1:2] - [1000.0, 0.0]).^2))
    # in_goal = dist <= 100.0
    # return [convert(Float64, in_goal)]
    return convert(Float64, norm(state.os[1:2] - SIM.goal_location) <= SIM.goal_radius)
end

function f_goal_dist(state::EncounterState, cutoff=nothing)
    if cutoff == nothing
        return max(0.0, norm(state.os[1:2] - SIM.goal_location)-SIM.goal_radius)
    else
        gd = max(0.0, norm(state.os[1:2] - SIM.goal_location)-SIM.goal_radius)
        if gd > cutoff
            return 0.0
        else
            return gd
        end
    end
end

function f_one(state::EncounterState)
    return convert(Float64, !state.end_state)
end

goal_dist_points = linspace(0.0, 500.0, 10)
goal_bearing_points = linspace(-pi, pi, 15)
const GOAL_GRID = RectangleGrid(goal_dist_points, goal_bearing_points)

intruder_dist_points = linspace(0.0, 700.0, 12) 
intruder_bearing_points = linspace(-pi/2, pi/2, 12)
intruder_heading_points = linspace(0.0, 2*pi, 12)
const INTRUDER_GRID = RectangleGrid(intruder_dist_points, intruder_bearing_points, intruder_heading_points)

features = [
    :f_in_goal,
    :f_goal_dist,
    :f_one,
    #XXX
    :f_has_deviated, # added 3/13
    ParameterizedFeatureFunction(:f_radial_goal_grid, GOAL_GRID, true),
    ParameterizedFeatureFunction(:f_focused_intruder_grid, INTRUDER_GRID, true),
    :f_conflict,
]

const FEATURES = FeatureBlock(features)

# function kaelbling_intruder(state)
#     return [dist(state), 1.0/dist(state), 
# end


# function return_param(p,s) p end
# 
# function test_module()
#     p = ParameterizedFeatureBlock([1,2], return_param)
#     b = assemble(p)
#     @test b.features(nothing) == [1,2]
#     b2 = AssembledFeatureBlock(b.description)
#     @test b2.features(nothing) == [1,2]
# end

end
