@info "Load Packages..."
using Pkg
Pkg.activate(@__DIR__)
using Profile
using POMDPModels
using POMDPs
using POMDPModelTools
using PFTDPW

@info "Instantiate Solver..."
tiger = TigerPOMDP()
solver = PFTDPWSolver(tree_queries=10_000, k_o=5, k_a=2, max_depth=10, c=100.0, n_particles=1_000, check_repeat_obs=true)
planner = solve(solver, tiger)
Profile.clear_malloc_data()

@info "Solve..."
a_info = action_info(planner, initialstate(tiger))

@info "Exit"
exit()
