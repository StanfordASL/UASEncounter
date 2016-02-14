import MDP
import MCTSdpw

using EncounterModel

import HRL
import PyPlot

initial_state = EncounterState([0.0,0.0,0.0],[500.0,-1000.0, pi/2.0],false)

ip = IntruderParams(60.0, 5.0/180.0*pi)
# ip = IntruderParams(60.0, 20.0/180.0*pi)
# ip = IntruderParams(60.0, 0.0)
op = OwnshipParams(30.0, 45.0/180.0*pi, 1.0, HRL.heading_hrl)
sim = SimParams(1.0, 9.8)
param = HRLParams(200.0)

rng_seed = 1
rng = MersenneTwister(rng_seed)

states = Array(EncounterState, 100)
# for i in 1:length(states)
#     states[i] = Array(Float64, length(initial_state))
# end

get_plot_x(state::AbstractArray{Float64,1}) = state[2]
get_plot_y(state::AbstractArray{Float64,1}) = state[1]

get_plot_u(heading::Float64) = cos(2*pi-(heading-pi/2))
get_plot_u(state::AbstractArray{Float64,1}) = get_plot_u(state[3])
get_plot_v(heading::Float64) = sin(2*pi-(heading-pi/2))
get_plot_v(state::AbstractArray{Float64,1}) = get_plot_v(state[3])

function plot(fig, sim::SimParams, op::OwnshipParams, ip::IntruderParams, state::EncounterState)
    if !state.end_state

        os = state.os
        is = state.is
        desired_heading = op.hrl(state, param, op, ip)
    
        ax = fig[:gca]()

        ax[:quiver](get_plot_x(os), get_plot_y(os), get_plot_u(desired_heading), get_plot_v(desired_heading))

        ax[:scatter](x=get_plot_x(os), y=get_plot_y(os), s=10, color="blue")
        ax[:scatter](x=get_plot_x(is), y=get_plot_y(is), s=10, color="red")
        ax[:set_xlim]((-600,600))
        ax[:set_ylim]((-100,1100))
    end
end

fig = PyPlot.figure(1)
fig[:clf]()
PyPlot.clf()
# plot(fig, sim, op, ip, initial_state)

states[1] = initial_state

for i in 2:length(states)
    # @show states
    states[i] = encounter_dynamics(sim, op, ip, states[i-1], param, rng)
end

constant_D_reward = 0.0
for i in 1:length(states)
    plot(fig, sim, op, ip, states[i])
    constant_D_reward += reward(op, ip, states[i], param)
    # @show states[i]
    # @show dist(states[i])
end
@show constant_D_reward

rng = MersenneTwister(rng_seed)
nSteps = length(states)
policy = MCTSdpw.selectAction

function getNextState(s::EncounterState,a::HRLParams,rng::AbstractRNG)
    return encounter_dynamics(sim, op, ip, s, a, rng)
end

function getInitialState(rng::AbstractRNG)
    return initial_state
end

function getReward(s::EncounterState, a::HRLParams)
    return reward(op, ip, s, a)
end

# default policy
function getAction(s::EncounterState, rng::AbstractRNG)
    return HRLParams(EncounterModel.LEGAL_D)
end

# exploration
function getNextAction(s::EncounterState, rng::AbstractRNG)
    possible = [50.0, 100.0, 150.0, 200.0, 300.0, 500.0]
    return HRLParams(possible[ceil(rand(rng)*length(possible))])
end

d = int16(100)           
ec = 100.
k = 5.
alpha = 0.5
kp = 5.
alphap = 0.5
n = int32(50) 

dpw = MCTSdpw.DPW(MCTSdpw.DPWParams(d,ec,n,k,alpha,kp,alphap,rng,getAction,getNextState,getReward,getNextAction))

encounter = MDP.GenerativeModel(getInitialState,getNextState,getReward,)

MDP.simulate(encounter,dpw,policy,nSteps,rng)
