using EncounterModel
using MCTSGlue
import HDF5, JLD
import Dates

lambdas = logspace(2,4,6)

policies = Array(Any, length(lambdas))

d = 70
ec = 100
k = 5
alpha = 0.5
kp = 5
alphap = 0.5
n = 50

lD = SIM.legal_D
actions = EncounterAction[HeadingHRL(D) for D in [0.0, lD, 1.5*lD, 2.0*lD, 3.0*lD, 4.0*lD]]

rng = MersenneTwister(0)

for i in 1:length(lambdas)
    rm = deepcopy(REWARD)
    rm.nmac_lambda = lambdas[i]
    policies[i] = MCTSPolicy(d, ec, k, alpha, kp, alphap, n, actions, rm, rng)
end

JLD.@save "../data/mcts_policies_$(Dates.format(Dates.now(),"u-d_HHMM")).jld" policies lambdas actions
