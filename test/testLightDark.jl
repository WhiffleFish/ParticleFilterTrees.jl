using QuickPOMDPs
using Distributions
using BenchmarkTools
using POMDPModelTools
using POMDPs
using PFTDPW
using D3Trees
include(join([@__DIR__,"/LightDarkPOMDP.jl"]))

solver = PFTDPWSolver(
    tree_queries=10_000,
    k_o=5,
    max_depth=30,
    c=100.0,
    n_particles=100,
    check_repeat_obs=false,
    enable_action_pw=false
)
planner = solve(solver, LightDark)

@benchmark a_info = action_info(planner, initialstate(LightDark)) (seconds=120)

@benchmark a_info = action_info(planner, initialstate(LightDark)) (seconds=120)

@profiler a_info = action_info(planner, initialstate(LightDark))

a, info = action_info(planner, initialstate(LightDark))
t = D3Tree(info[:tree])
inchrome(t)
