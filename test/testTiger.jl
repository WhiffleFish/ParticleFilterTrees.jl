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

solver = PFTDPWSolver(max_time=0.1, tree_queries=100_000, k_o=1, k_a=2, max_depth=10, c=100.0, n_particles=100)
planner = solve(solver, tiger)

a,info = action_info(planner, initialstate(tiger))
a = action(planner, initialstate(tiger))

##
b = initial_belief(initialstate(tiger), 100)
f(p,s,b) = PFTDPW.rollout(p, s, b, 10)
@code_warntype f(planner, solver, b)
