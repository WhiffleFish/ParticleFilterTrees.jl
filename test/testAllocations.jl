println("Load Packages...")
using Pkg
using Profile
using POMDPModels
Pkg.activate(".")
include("../src/PFTDPW.jl")

println("Instantiate Solver...")
tiger = TigerPOMDP()
solver = PFTDPWSolver(tree_queries=10_000, k_o=1, k_a=2, max_depth=10, c=100.0, n_particles=1_000)
planner = solve(tiger, solver)
Profile.clear_malloc_data()

println("Solve...")
a_info = action_info(planner, initialstate(tiger))

println("Exit")
exit()
