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
    enable_action_pw=false,
    resample=false
)
planner = solve(solver, LightDark)

@benchmark a_info = action_info(planner, initialstate(LightDark)) (seconds=120)

@benchmark a_info = action_info(planner, initialstate(LightDark)) (seconds=120)

@benchmark action_info(planner, initialstate(LightDark))

@benchmark action_info(planner, initialstate(LightDark))

@profiler a_info = action_info(planner, initialstate(LightDark))

a, info = action_info(planner, initialstate(LightDark))
t = D3Tree(info[:tree])
inchrome(t)
tree = info[:tree]


##
using POMCPOW
pomcpow_solver = POMCPOWSolver(
    tree_queries = 10_000,
    k_observation=5,
    check_repeat_obs=false,
    max_depth=30,
    criterion=MaxUCB(100.0),
)
pomcpow_planner = solve(pomcpow_solver, LightDark)

action_info(pomcpow_planner, initialstate(LightDark))

@profiler action_info(pomcpow_planner, initialstate(LightDark))

@benchmark action_info(pomcpow_planner, initialstate(LightDark))

a, info = action_info(pomcpow_planner, initialstate(LightDark))

pomcpow_planner.tree

info[:tree].b[2000]
