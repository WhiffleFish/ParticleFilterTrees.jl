using POMDPs
using POMDPModels
using ProgressMeter
using POMDPSimulators
using BeliefUpdaters

using PFTDPW
pomdp = TigerPOMDP()

t = 0.1
d = 10
pft_solver = PFTDPWSolver(max_time=t, tree_queries=100_000, k_o=5, alpha_o=0.1,k_a=2, max_depth=d, c=100.0, n_particles=100)
pft_planner = solve(pft_solver, pomdp)

function benchmark(pomdp::POMDP, planner::Policy; depth::Int=20, N::Int=100)
    r1Hist = Float64[]
    ro = RolloutSimulator(max_steps=depth)
    upd = DiscreteUpdater(pomdp)
    @showprogress for i = 1:N
        r1 = simulate(ro, pomdp, planner, upd)
        push!(r1Hist, r1)
    end
    return r1Hist::Vector{Float64}
end

N = 100
r_pft = benchmark(pomdp, pft_planner, N=N)
mean(r_pft)
