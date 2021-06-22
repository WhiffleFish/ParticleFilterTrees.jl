using POMDPs
using POMDPModels
using POMDPSimulators
using ParticleFilters
using POMCPOW
using ProgressMeter
using BenchmarkTools
using Plots
using Statistics

using PFTDPW
using QuickPOMDPs
using POMDPModelTools
using Distributions

const R = 60
const LIGHT_LOC = 10

const LightDark = QuickPOMDP(
    states = -R:R+1,                  # r+1 is a terminal state
    actions = [-10, -1, 0, 1, 10],
    discount = 0.95,
    isterminal = s::Int -> s==R::Int+1,
    obstype = Float64,

    transition = function (s::Int, a::Int)
        if a == 0
            return Deterministic{Int}(R::Int+1)
        else
            return Deterministic{Int}(clamp(s+a, -R::Int, R::Int))
        end
    end,

    observation = (s, a, sp) -> Normal(sp, abs(sp - LIGHT_LOC::Int) + 1e-3),

    reward = function (s, a, sp, o)
        if a == 0
            return s == 0 ? 100 : -100
        else
            return -1.0
        end
    end,

    initialstate = POMDPModelTools.Uniform(div(-R::Int,2):div(R::Int,2))
)

t = 0.1
d = 50
pft_solver = PFTDPWSolver(
    max_time=t,
    tree_queries=100_000,
    k_o=10.0,
    alpha_o=1/20,
    k_a=4,
    max_depth=d,
    c=100.0,
    n_particles=100,
    enable_action_pw = false
)

pft_planner = solve(pft_solver, LightDark)

pomcpow_solver = POMCPOWSolver(
    max_time=t,
    tree_queries = 10_000_000,
    max_depth=d,
    criterion = MaxUCB(95.0),
    tree_in_info=false,
    enable_action_pw = false
)
pomcpow_planner = solve(pomcpow_solver, LightDark)

function benchmark(pomdp::POMDP, planner1::Policy, planner2::Policy; depth::Int=20, N::Int=100)
    r1Hist = sizehint!(Float64[],N)
    r2Hist = sizehint!(Float64[],N)
    ro = RolloutSimulator(max_steps=depth)
    upd = BootstrapFilter(pomdp, 1_000)
    @showprogress for i = 1:N
        r1 = simulate(ro, pomdp, planner1, upd)
        r2 = simulate(ro, pomdp, planner2, upd)
        push!(r1Hist, r1)
        push!(r2Hist, r2)
    end
    return (r1Hist, r2Hist)::Tuple{Vector{Float64},Vector{Float64}}
end

N = 100
r_pft, r_pomcp = benchmark(LightDark, pft_planner, pomcpow_planner, N=N, depth=d)

histogram([r_pft r_pomcp], alpha=0.5, labels=["PFT-DPW" "POMCPOW"], normalize=true, legend=:topright)
title!("LightDark1D Benchmark\nt=$(t)s, d=$d, N=$N")
xlabel!("Returns")
ylabel!("Density")
mean(r_pft)
mean(r_pomcp)
std(r_pft)/sqrt(N)
std(r_pomcp)/sqrt(N)
