using ArgParse
import HDF5, JLD
using EncounterFeatures
using EncounterSimulation
import EncounterVisualization
import PyPlot

s = ArgParseSettings()

@add_arg_table s begin
    "filename"
        help = "filename containing policy"
        required = true
    "policy_name"
        help = "dictionary key for policy"
        default = "policy"
    "--thresh"
        arg_type = Float64
        default = 0.0
    "--vals"
        action = :store_true
end

args = parse_args(ARGS, s)

println("using file $(args["filename"])")

data = JLD.load(args["filename"])

policy = extract_from_record(data[args["policy_name"]])

oad = 0.0
iad = 135.0
ix = 600.0
iy = -250.0
EncounterVisualization.plot_policy_grid(policy,
                                        [ix, iy, pi/180.0*iad],
                                        pi/180.0*oad,
                                        100,
                                        threshold=args["thresh"])

if args["vals"]
    for a in 1:length(policy.actions)
        PyPlot.figure(a+1)
        EncounterVisualization.plot_value_grid(policy.phi, policy.thetas[a], [ix, iy, pi/180.0*iad], pi/180.0*oad)
        PyPlot.title(string(policy.actions[a]))
    end
end

println("press enter to exit")
readline(STDIN)
