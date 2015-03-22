using EncounterModel
using EncounterFeatures
using EncounterValueIteration

# make sure to uncomment test lines

rnga = MersenneTwister(0)
rngb = MersenneTwister(0)
for i = 1:100000
    gen_state_snap_to_grid(rnga, INTRUDER_GRID, GOAL_GRID)
    # @show state = gen_state_snap_to_grid(rng, INTRUDER_GRID, GOAL_GRID)
    # fi = f_focused_intruder_grid(state, INTRUDER_GRID)
    # fg = f_radial_goal_grid(state, GOAL_GRID)
    # 
    # if sum(fi) > 1e-5 
    #     @assert maximum(fi) >= 0.95
    # end
    # if sum(fg) > 1e-5
    #     @assert maximum(fg) >= 0.95
    # end
end
