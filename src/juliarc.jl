println("executing local juliarc")

@everywhere mdp_path = "$(Pkg.dir())/mdp/"
@everywhere include("$(mdp_path)src/MDP.jl")
@everywhere include("$(mdp_path)src/mdps/GridWorld.jl")
@everywhere include("$(mdp_path)src/other/auxfuncs.jl")
# include("$(mdp_path)src/solvers/MCTSdpw/AG.jl")
@everywhere include("$(mdp_path)src/solvers/MCTSdpw/MCTSdpw.jl")
