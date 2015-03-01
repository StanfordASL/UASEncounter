

module EncounterFeatures

using EncounterModel: EncounterState, SIM, dist, mindist, toca
using GridInterpolations

using LegacyFeatures
import LegacyFeatures.f_radial_goal_grid

import Base.Test.@test

export ParameterizedFeatureBlock, ParameterizedFeatureFunction
export assemble, test_module
export AssembledFeatureBlock
export f_in_goal, f_mindist_time, f_one_over_mindist_time, f_exp_neg_mindist, f_exp_neg_dist, f_one_over_mindist, f_one_over_dist, f_intruder_dist, f_one, f_goal_dist, f_abs_goal_bearing, f_radial_intruder_grid, f_radial_goal_grid, f_exp_neg_goal_dist, f_within_goal_dist, f_conflict, f_focused_intruder_grid, f_half_intruder_bin_grid

abstract FeatureBlock
function Base.showall(io::IO, block::FeatureBlock)
    show(io, typeof(block))
    print(io, '(')
    for i in 1:length(names(block))
        print(io, repr(getfield(block, i)))
        print(io, ',')
    end
    print(io, ')')
end

# fun returns a single element vector, there are many param values
type ParameterizedFeatureBlock <: FeatureBlock
    fun::Function
    params::Vector{Any}
end

# fun may return a multi-element vector, there is only one param value
type ParameterizedFeatureFunction <: FeatureBlock
    fun::Function
    param::Any
    memory::Bool
end
ParameterizedFeatureFunction(fun::Function, param::Any) = ParameterizedFeatureFunction(fun, param, false)

type AssembledFeatureBlock
    length::Int
    features::Function
    dense::Bool
    memory::Bool
    description::Vector{String}
end
function AssembledFeatureBlock{T<:String}(features::Function, description::Vector{T}; memory=false)
    f = features(gen_test_state())
    # @assert isa(f, AbstractArray{Float64}) 
    AssembledFeatureBlock(length(f), features, isa(f,Vector{Float64}) || isa(f,Float64), memory, description)
end
function AssembledFeatureBlock{T<:String}(description::Vector{T})
    blocks = Any[]
    for d in description
        push!(blocks, eval(parse(d)))
    end
    return assemble(blocks)
end

function assemble(fun::Function)
    return AssembledFeatureBlock(fun, [repr(fun)])
end

function assemble(block::ParameterizedFeatureBlock)
    fun(state) = call_with_params(block.fun, block.params, state)
    return AssembledFeatureBlock(fun, [repr(block)])
end

function assemble(block::ParameterizedFeatureFunction)
    if block.memory
        fun(state; memory=Float64[]) = block.fun(state, block.param, memory=memory)
    else
        fun(state) = block.fun(state, block.param)
    end
    return AssembledFeatureBlock(fun, [repr(block)], memory=block.memory)
end

function assemble(block::AssembledFeatureBlock)
    return block
end

function assemble{T}(blocks::Vector{T})
    length = 0
    assembled = AssembledFeatureBlock[]
    descriptions = String[]
    for i in 1:size(blocks)[1]
        b = assemble(blocks[i])
        length += b.length
        append!(descriptions, b.description)
        push!(assembled, b)
    end
    dense = all([a.dense for a in assembled])
    needs_memory = any([a.memory for a in assembled])
    return AssembledFeatureBlock(length, s -> call_all(assembled, s), dense, needs_memory, descriptions)
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
    if heading >= 2*pi heading -= 2*pi end
    if heading < 0.0 heading += 2*pi end

    if bearing > pi/2 || bearing < -pi/2 || d <= 0.0
        return phi
    end

    inds, weights = interpolants(grid, [d, bearing, heading])

    for i in 1:length(inds)
        phi[inds[i]] = weights[i]
    end
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
    while bearing < 0.0 bearing += 2*pi end
    while bearing >= 2*pi bearing -= 2*pi end

    inds, weights = interpolants(grid, [d, bearing])

    for i in 1:length(inds)
        phi[inds[i]] = weights[i]
    end
    return phi
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

function call_with_params(fun::Function, params::Vector{Any}, state::EncounterState)
    b = Array(Float64, length(params))
    for i in 1:length(params)
        b[i:i] = fun(state, params[i])
    end
    return b
end

function call_all(blocks::Vector{AssembledFeatureBlock}, state::EncounterState; memory::AbstractVector{Float64}=Float64[])
    b = memory
    if length(memory)==0 && all([a.dense for a in blocks])
        b = zeros(Float64, sum([a.length for a in blocks]))
        # b = Array(Float64, sum([a.length for a in blocks]))
    elseif length(memory)==0
        b = spzeros(sum([a.length for a in blocks]),1)
    end
    i = 1
    for a in blocks
        if a.memory
            a.features(state, memory=sub(b, i:i-1+a.length))
        else
            b[i:i-1+a.length] = a.features(state)
        end
        i+=a.length
    end
    return b
end

function gen_test_state()
    return EncounterState([0.,0.,0.], [0.,0.,0.],true)
end

# function kaelbling_intruder(state)
#     return [dist(state), 1.0/dist(state), 
# end


function return_param(p,s) p end

function test_module()
    p = ParameterizedFeatureBlock([1,2], return_param)
    b = assemble(p)
    @test b.features(nothing) == [1,2]
    b2 = AssembledFeatureBlock(b.description)
    @test b2.features(nothing) == [1,2]
end

end
