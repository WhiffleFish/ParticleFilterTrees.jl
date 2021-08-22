@info "Load Packages..."
using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
include(joinpath(@__DIR__,"../LightDarkPOMDP.jl"))
using Profile
using POMDPModels
using POMDPs
using POMDPModelTools
using PFTDPW

@info "Instantiate Solver..."
pomdp = LightDark
solver = PFTDPWSolver(
    tree_queries=100_000,
    k_o=10,
    max_depth=50,
    c=100.0,
    n_particles=20,
    check_repeat_obs=false,
    enable_action_pw = false
)
planner = solve(solver, pomdp)
Profile.clear_malloc_data()

@info "Solve..."
a = action(planner, initialstate(pomdp))

@info "Exit"
exit()
