using ArgParse
import HDF5, JLD
using EncounterFeatures
import EncounterVisualization
import PyPlot

s = ArgParseSettings()

@add_arg_table s begin
    "filename"
        help = "filename containing policy"
        required = true
end

parsed_args = parse_args(ARGS, s)

println("using file $(parsed_args["filename"])")

data = JLD.load(parsed_args["filename"])

# phi = AssembledFeatureBlock(data["phi_description"])
phi = data["phi"]

oad = 0.0
iad = 135.0
ix = 700.0
iy = -250.0
EncounterVisualization.plot_value_grid(phi,
                                       data["theta"],
                                       [ix, iy, pi/180.0*iad],
                                       pi/180.0*oad,
                                       100)

PyPlot.show()

# println("press enter to exit")
# readline(STDIN)
