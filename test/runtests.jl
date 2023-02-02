using Pkg
# Pkg.add(url = "https://github.com/JuliaPOMDP/LaserTag.jl#master")
Pkg.add(url = "https://github.com/WhiffleFish/SubHunt.jl#master")
Pkg.add(url = "https://github.com/WhiffleFish/VDPTag2.jl#master")

using Test
using POMDPTools
using POMDPs, POMDPModels, QuickPOMDPs
using QMDP
using Random
using StaticArrays
using ParticleFilters
using ParticleFilterTrees
using SubHunt
using D3Trees
# using LaserTag
using VDPTag2

include("LightDarkPOMDP.jl")

include("PFTDPW.jl")

include("subhunt_terminal.jl")

include("sampling.jl")
