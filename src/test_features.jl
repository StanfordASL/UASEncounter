using EncounterModel
using EncounterFeatures
using Base.Test

f = ParameterizedFeatureFunction(:f_focused_intruder_grid,INTRUDER_GRID)
fg = ParameterizedFeatureFunction(:f_symmetric_goal_grid,GOAL_GRID)

s = EncounterState([0,0,0],[0,0,0],true,false)
memory = ones(length(f))
@test all(evaluate(f,s,memory=memory).==0.0)
memory = ones(length(f))
@test all(evaluate(fg,s,memory=memory).==0.0)


xi = 1.2*maximum(INTRUDER_GRID.cutPoints[1])
s = EncounterState([0,0,0],[xi, 0, pi],false,false)
@test all(evaluate(f,s).==0.0)
@test all(evaluate(fg,s).==0.0)
