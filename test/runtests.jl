using Pkg
# Pkg.add(url = "https://github.com/JuliaPOMDP/LaserTag.jl#master")
Pkg.add(url = "https://github.com/WhiffleFish/SubHunt.jl#master")
Pkg.add(url = "https://github.com/zsunberg/VDPTag2.jl#master")

using Test
using POMDPs, POMDPModelTools, POMDPModels, QuickPOMDPs, POMDPSimulators
using QMDP
using Random
using StaticArrays
using ParticleFilters
using ParticleFilterTrees
using SubHunt
# using LaserTag
using VDPTag2

include("LightDarkPOMDP.jl")

include("PFTDPW.jl")

include("subhunt_terminal.jl")

include("sampling.jl")
