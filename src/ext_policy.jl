import HDF5
import JLD

using EncounterFeatures
using EncounterModel
@everywhere using EncounterValueIteration
using EncounterSimulation
@everywhere using GridInterpolations
using ArgParse

s = ArgParseSettings()

@add_arg_table s begin
    "input"
        required = true
    "output"
        required = true
end

args=parse_args(ARGS, s)

try
    rm("/tmp/current_val")
catch(e)
    println(e)
    println("continuing anyways...")
end
# run(`ln -s $full_filepath /tmp/current_val`)
symlink(abspath(args["input"]),"/tmp/current_val")

@everywhere begin
    data = JLD.load("/tmp/current_val")
    intruder_grid = data["intruder_grid"]
    snap_generator(rng) = gen_state_snap_to_grid(rng, intruder_grid)
end

rng0 = MersenneTwister(0)
ic_batch = gen_ic_batch_for_grid(rng0, intruder_grid)

# phi = AssembledFeatureBlock(data["phi_description"])
phi = data["phi"]
theta = data["theta"]
actions = data["actions"]

policy = extract_policy(phi,
                        theta,
                        actions,
                        50000,
                        ic_batch=ic_batch,
                        state_gen=snap_generator)


JLD.save(string(args["output"]), "policy", make_record(policy))

# XXX HACK HACK HACK
# @everywhere begin
#     intruder_dist_points = linspace(0.0, 700.0, 12) 
#     intruder_bearing_points = linspace(-pi/2, pi/2, 12)
#     intruder_heading_points = linspace(0.0, 2*pi, 12)
#     intruder_grid = RectangleGrid(intruder_dist_points, intruder_bearing_points, intruder_heading_points)
# end
# 
# @everywhere snap_generator(rng) = gen_state_snap_to_grid(rng, intruder_grid)
# rng0 = MersenneTwister(0)
# ic_batch = gen_ic_batch_for_grid(rng0, intruder_grid)
# 
# hrl_data = JLD.load("../data/hrl_val.jld")
# 
# hrl_phi = AssembledFeatureBlock(hrl_data["phi_description"])
# hrl_lambda = hrl_data["lambda"]
# 
# lD = SIM.legal_D
# hrl_actions = EncounterAction[HeadingHRL(D) for D in [lD, 1.1*lD, 1.2*lD, 1.5*lD, 2.0*lD]]
# 
# hrl_policy = extract_policy(hrl_phi, hrl_lambda, hrl_actions, 10000, ic_batch=ic_batch, state_gen=snap_generator)
# 
# bank_data = JLD.load("../data/bank_val.jld")
# 
# bank_phi = AssembledFeatureBlock(bank_data["phi_description"])
# bank_lambda = bank_data["lambda"]
# 
# bank_actions = EncounterAction[BankControl(b) for b in [-OWNSHIP.max_phi, -OWNSHIP.max_phi/2, 0.0, OWNSHIP.max_phi/2, OWNSHIP.max_phi]]
# 
# bank_policy = extract_policy(bank_phi, bank_lambda, bank_actions, 10000, ic_batch=ic_batch, state_gen=snap_generator)
# 
# JLD.save("../data/nice_pol.jld", "hrl_policy", make_record(hrl_policy), "bank_policy", make_record(bank_policy))
