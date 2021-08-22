using POMDPs
using D3Trees
using ParticleFilters
using POMDPModels
using POMDPModelTools
using BenchmarkTools
using PFTDPW
using JET
using JETTest

tiger = TigerPOMDP()

solver = PFTDPWSolver(
    tree_queries=10_000,
    k_o=5,
    max_depth=10,
    c=100.0,
    n_particles=1000,
    check_repeat_obs=true,
    enable_action_pw=false
)
planner = solve(solver, tiger)
action(planner, initialstate(tiger))

##

@report_dispatch action(planner, initialstate(tiger))

@report_call action(planner, initialstate(tiger))

@benchmark action(planner, initialstate(tiger))

@benchmark action(planner, initialstate(tiger)) (seconds=120)

@profiler a,info = action_info(planner, initialstate(tiger)) recur=:flat
a,info = action_info(planner, initialstate(tiger))
inchrome(D3Tree(info[:tree]))
