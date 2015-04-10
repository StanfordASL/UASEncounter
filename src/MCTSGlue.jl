module MCTSGlue

mdp_path = "/home/zach/.julia/mdp/"
include("$(mdp_path)src/MDP.jl")
include("$(mdp_path)src/mdps/GridWorld.jl")
include("$(mdp_path)src/other/auxfuncs.jl")
# include("$(mdp_path)src/solvers/MCTSdpw/AG.jl")
include("$(mdp_path)src/solvers/MCTSdpw/MCTSdpw.jl")

import MDP
import MCTSdpw
import GridWorld

using EncounterTestModel

type MDPEncounterState <: MDP.State
    s::EncounterState
end

type MDPEncounterAction <: MDP.Action
    a::EncounterAction
end



end
