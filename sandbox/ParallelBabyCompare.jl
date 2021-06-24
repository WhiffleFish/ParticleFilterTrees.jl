using POMDPs
using BeliefUpdaters
using POMDPModels
using Plots
using Statistics
using Distributed
include(join([@__DIR__,"/benchmark.jl"]))
baby = BabyPOMDP()
worker_ids = Distributed.addprocs(40, topology=:master_worker, exeflags="--project=./test")

@everywhere begin
    using Pkg
    Pkg.instantiate()
    using ProgressMeter
    using POMDPSimulators
    using PFTDPW
    using POMCPOW
    using POMDPModels
end

t = 0.1
d=50
pft_solver = PFTDPWSolver(
    max_time=t,
    tree_queries=100_000,
    k_o = 10,
    k_a = 2,
    max_depth = d,
    c = 100.0,
    n_particles = 100,
    enable_action_pw = false,
    check_repeat_obs = true
)
pft_planner = solve(pft_solver, baby)

pomcpow_solver = POMCPOWSolver(
    max_time=t,
    tree_queries = 1_000_000,
    max_depth=d,
    criterion = MaxUCB(100.0),
    tree_in_info=false,
    enable_action_pw=false
)
pomcpow_planner = solve(pomcpow_solver, baby)

N = 500
r_pft, r_pomcp = benchmark(
    baby,
    DiscreteUpdater(baby),
    [pft_planner, pomcpow_planner],
    N,
    d
)

Distributed.rmprocs(worker_ids)

histogram([r_pft r_pomcp], alpha=0.5, labels=["PFT-DPW" "POMCPOW"], normalize=true, legend=:topleft)
title!("Baby Benchmark\nt=$(t)s, d=$d, N=$N")
xlabel!("Returns")
ylabel!("Density")
mean(r_pft)
mean(r_pomcp)
std(r_pft)/sqrt(N)
std(r_pomcp)/sqrt(N)
