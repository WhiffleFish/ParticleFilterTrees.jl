using POMDPModels
using POMDPModelTools
using POMCPOW
using BenchmarkTools

pomdp = LightDark1D()
include("../src/PFTDPW.jl")
include("../src/tree_visualization.jl")

solver = PFTDPWSolver(tree_queries=10_000, k_o=1, k_a=2, max_depth=10, c=100.0, n_particles=1000, check_repeat_obs=false)
planner = solve(pomdp, solver)
@benchmark a_info = action_info(planner, initialstate(pomdp)) (seconds=120)

@benchmark a_info = action_info(planner, initialstate(pomdp)) (seconds=120)

@profiler a_info = action_info(planner, initialstate(pomdp))

##
pomcpow_solver = POMCPOWSolver(
    max_time=0.1,
    tree_queries = 100_000,
    max_depth=10,
    criterion = MaxUCB(100.0),
    tree_in_info=false,
    enable_action_pw = false,
    k_observation = 1
)

pomcpow_planner = solve(pomcpow_solver, tiger)

a, a_inf = action_info(pomcpow_planner, initialstate(tiger))

@benchmark action_info(planner, initialstate(tiger)) (seconds=60)





@benchmark action_info(pomcpow_planner, initialstate(tiger)) (seconds=60)

inchrome(D3Tree(a_inf[:tree]))

@benchmark action_info(pomcpow_planner, initialstate(tiger))
