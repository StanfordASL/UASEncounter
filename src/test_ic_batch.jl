using EncounterModel
using EncounterFeatures
using EncounterValueIteration

# make sure to uncomment test lines

rnga = MersenneTwister(0)
# rngb = MersenneTwister(0)
uc = nothing
dc = nothing
for i = 1:1000
    print(".")
    b = gen_ic_batch_for_grid(rnga, INTRUDER_GRID, GOAL_GRID)
    intruder_inds = zeros(length(INTRUDER_GRID))
    goal_inds = zeros(length(GOAL_GRID))
    for j in 1:length(b)
        s = b[j]

        fi = f_focused_intruder_grid(s, INTRUDER_GRID)
        intruder_inds[indmax(fi)] +=1

        gi = f_symmetric_goal_grid(s, GOAL_GRID)
        goal_inds[indmax(gi)] += 1
    end
    if any(intruder_inds.<0.95)
        @show intruder_inds
        @show dc = find(intruder_inds.>1.05)
        @show uc = find(intruder_inds.<0.95)
        @assert all(intruder_inds.>=0.95)
    end
    if any(goal_inds.<0.95)
        @show goal_inds
        @assert all(goal_inds.>=0.95)
    end
end
