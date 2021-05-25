using POMDPs
using POMDPModels
using POMCPOW
using ProgressMeter
using BenchmarkTools
using POMDPSimulators
using Plots
using BeliefUpdaters

using PFTDPW
pomdp = TigerPOMDP()

t = 0.1
d=10
pft_solver = PFTDPWSolver(max_time=t, tree_queries=100_000, k_o=1, k_a=2, max_depth=d, c=100.0, n_particles=100)
pft_planner = solve(pomdp, pft_solver)

pomcpow_solver = POMCPOWSolver(
    max_time=t,
    tree_queries = 100_000,
    max_depth=d,
    criterion = MaxUCB(100.0),
    enable_action_pw=false,
    k_action=2,
    tree_in_info=false)
pomcpow_planner = solve(pomcpow_solver, pomdp)

function benchmark(pomdp::POMDP, planner1::Policy, planner2::Policy; depth::Int=20, N::Int=100)
    r1Hist = Float64[]
    r2Hist = Float64[]
    ro = RolloutSimulator(max_steps=depth)
    upd = DiscreteUpdater(pomdp)
    @showprogress for i = 1:N
        r1 = simulate(ro, pomdp, planner1, upd)
        r2 = simulate(ro, pomdp, planner2, upd)
        push!(r1Hist, r1)
        push!(r2Hist, r2)
    end
    return (r1Hist, r2Hist)::Tuple{Vector{Float64},Vector{Float64}}
end

N = 100
r_pft, r_pomcp = benchmark(pomdp, pft_planner, pomcpow_planner, N=N)

histogram([r_pft r_pomcp], alpha=0.5, labels=["PFT-DPW" "POMCPOW"], normalize=true, bins=20, legend=:topleft)
title!("Tiger Benchmark\nt=$(t)s, d=$d, N=$N")
xlabel!("Returns")
ylabel!("Density")
mean(r_pft)
mean(r_pomcp)
