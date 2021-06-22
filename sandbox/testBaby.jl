using POMDPModels
using POMDPModelTools
using BenchmarkTools
using POMDPs
using PFTDPW

baby = BabyPOMDP()

solver = PFTDPWSolver(tree_queries=100_000, k_o=1, k_a=2, max_depth=10, c=100.0, n_particles=1000)
planner = solve(solver, baby)
@benchmark a_info = action_info(planner, initialstate(baby)) (seconds=60)

@benchmark a_info = action_info(planner, initialstate(baby)) (seconds=60)

@profiler a_info = action_info(planner, initialstate(baby))

a,info = action_info(planner, initialstate(baby))
t = D3Tree(info[:tree])
inchrome(t)

info[:tree].bao_children
