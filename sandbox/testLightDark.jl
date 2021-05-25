using POMDPModels
using BenchmarkTools
using POMDPModelTools
using POMDPs

using PFTDPW
pomdp = LightDark1D()


solver = PFTDPWSolver(tree_queries=10_000, k_o=1, k_a=2, max_depth=10, c=100.0, n_particles=1000, check_repeat_obs=false)
planner = solve(solver, pomdp)
@benchmark a_info = action_info(planner, initialstate(pomdp)) (seconds=120)

@benchmark a_info = action_info(planner, initialstate(pomdp)) (seconds=120)

@profiler a_info = action_info(planner, initialstate(pomdp))
