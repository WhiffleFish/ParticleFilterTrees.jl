using QuickPOMDPs
using Distributions
using BenchmarkTools
using POMDPModelTools
using POMDPs
using PFTDPW
using D3Trees
using JET
using JETTest
include(joinpath(@__DIR__,"../LightDarkPOMDP.jl"))

using QMDP
VE = PFTDPW.PORollout(QMDPSolver(), n_rollouts=10)
solver = PFTDPWSolver(
    tree_queries=10_000,
    k_o=5,
    max_depth=30,
    c=100.0,
    n_particles=100,
    check_repeat_obs=false,
    enable_action_pw=false,
)
planner = solve(solver, LightDark)
action(planner, initialstate(LightDark))

@report_dispatch action(planner, initialstate(LightDark))

@benchmark action(planner, $(initialstate(LightDark)))

@benchmark action(planner, initialstate(LightDark)) (seconds=120)

@profiler action(planner, initialstate(LightDark)) recur=:flat

a, info = action_info(planner, initialstate(LightDark))
t = D3Tree(info[:tree])
inchrome(t)

using BasicPOMCP
using DiscreteValueIteration
VE = FOValue(ValueIterationSolver())
solver = PFTDPWSolver(
    tree_queries=10_000,
    k_o=5,
    max_depth=30,
    c=100.0,
    value_estimator = VE,
    n_particles=1000,
    check_repeat_obs=true,
    enable_action_pw=false,
)
planner = solve(solver, LightDark)
a, info = action_info(planner, initialstate(LightDark))
t = D3Tree(info[:tree])
inchrome(t)
