@info "Load Packages..."
using Pkg
Pkg.activate(".")
using Profile
using POMDPModels
using PFTDPW

@info "Instantiate Solver..."
tiger = TigerPOMDP()
solver = PFTDPWSolver(tree_queries=10_000, k_o=1, k_a=2, max_depth=10, c=100.0, n_particles=1_000)
planner = solve(solver, tiger)
Profile.clear_malloc_data()

@info "Solve..."
a_info = action_info(planner, initialstate(tiger))

@info "Exit"
exit()
