using POMDPModels
using POMDPModelTools
using POMCPOW
using BenchmarkTools
# using Plots

#=
const TIGER_LISTEN = 0
const TIGER_OPEN_LEFT = 1
const TIGER_OPEN_RIGHT = 2

const TIGER_LEFT = true
const TIGER_RIGHT = false
=#

tiger = TigerPOMDP()
include("../src/PFTDPW.jl")
include("../src/tree_visualization.jl")

solver = PFTDPWSolver(tree_queries=10_000, k_o=1, k_a=2, max_depth=10, c=100.0, n_particles=1_000)
planner = solve(tiger, solver)
@benchmark  a_info = action_info(planner, initialstate(tiger)) (seconds=60)


a_info = action_info(planner, initialstate(tiger))

tree = a_info[:tree]
d3t = D3Tree(tree)
inchrome(d3t)

##
pomcpow_solver = POMCPOWSolver(tree_queries = 10_000, max_depth=10, criterion = MaxUCB(100.0), tree_in_info=true)
pomcpow_planner = solve(pomcpow_solver, tiger)

a, a_inf = action_info(pomcpow_planner, initialstate(tiger))

inchrome(D3Tree(a_inf[:tree]))
