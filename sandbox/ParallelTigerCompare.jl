using Plots
using BeliefUpdaters
using Statistics
using Distributed

worker_ids = Distributed.addprocs(20; exeflags="--project=./test")
@everywhere begin
    using PFTDPW
    using POMCPOW
    using POMDPModels
    using POMDPs
    using POMDPSimulators
    tiger = TigerPOMDP()
end
include(join([@__DIR__,"/benchmark.jl"]))

t = 0.1
d=10
pft_solver = PFTDPWSolver(
    max_time=t,
    tree_queries=100_000,
    k_o=1,
    k_a=2,
    max_depth=d,
    c=100.0,
    n_particles=100,
    enable_action_pw=false,
    check_repeat_obs=true
)
pft_planner = solve(pft_solver, tiger)

pomcpow_solver = POMCPOWSolver(
    max_time=t,
    tree_queries = 100_000,
    max_depth=d,
    criterion = MaxUCB(100.0),
    enable_action_pw=false,
    k_action=2,
    tree_in_info=false)
pomcpow_planner = solve(pomcpow_solver, tiger)

N = 500
r_pft, r_pomcp = benchmark(
    tiger,
    DiscreteUpdater(tiger),
    [pft_planner, pomcpow_planner],
    N,
    d
)

Distributed.rmprocs(worker_ids)

histogram([r_pft r_pomcp], alpha=0.5, labels=["PFT-DPW" "POMCPOW"], normalize=true, legend=:topleft)
title!("Tiger Benchmark\nt=$(t)s, d=$d, N=$N")
xlabel!("Returns")
ylabel!("Density")
mean(r_pft)
mean(r_pomcp)
std(r_pft)/N
std(r_pomcp)/N
