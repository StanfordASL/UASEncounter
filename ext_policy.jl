import HDF5
import JLD

using EncounterFeatures: AssembledFeatureBlock
using EncounterModel
using EncounterValueIteration
using EncounterSimulation
using GridInterpolations

hrl_data = JLD.load("../data/hrl_val.jld")

hrl_phi = AssembledFeatureBlock(hrl_data["phi_description"])
hrl_lambda = hrl_data["lambda"]

lD = SIM.legal_D
hrl_actions = EncounterAction[HeadingHRL(D) for D in [lD, 1.1*lD, 1.2*lD, 1.5*lD, 2.0*lD]]

hrl_policy = extract_policy(hrl_phi, hrl_lambda, hrl_actions, 50000)

accel_data = JLD.load("../data/accel_val.jld")

accel_phi = AssembledFeatureBlock(accel_data["phi_description"])
accel_lambda = accel_data["lambda"]


accel_actions = EncounterAction[BankControl(b) for b in [-OWNSHIP.max_phi, -OWNSHIP.max_phi/2, 0.0, OWNSHIP.max_phi/2, OWNSHIP.max_phi]]

accel_policy = extract_policy(accel_phi, accel_lambda, accel_actions, 50000)

JLD.save("../data/first_pol.jld", "hrl_policy", make_record(hrl_policy), "accel_policy", make_record(accel_policy))
