using POMDPs
using ParticleFilters
using POMDPModels
using POMDPModelTools
using BenchmarkTools
using PFTDPW

tiger = TigerPOMDP()

solver = PFTDPWSolver(tree_queries=10_000, k_o=1, k_a=2, max_depth=10, c=100.0, n_particles=1000)
planner = solve(solver, tiger)
@benchmark a_info = action_info(planner, initialstate(tiger)) (seconds=120)

@benchmark a_info = action_info(planner, initialstate(tiger)) (seconds=120)

@profiler a,info = action_info(planner, initialstate(tiger))
