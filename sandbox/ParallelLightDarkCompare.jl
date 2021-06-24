using POMDPs
using ParticleFilters
using Plots
using Statistics
using Distributed
using Distributions
using Random

worker_ids = Distributed.addprocs(20; exeflags="--project=./test")
@everywhere begin
    using Pkg
    Pkg.instantiate()
    using ProgressMeter
    using QuickPOMDPs
    using POMDPSimulators
    using POMDPModelTools
    using PFTDPW
    using POMCPOW

    using Distributions
    const R = 60
    const LIGHT_LOC = 10

    const LightDark = QuickPOMDP(
        states = -R:R+1, # r+1 is a terminal state
        actions = (-10, -1, 0, 1, 10),
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

end

include(join([@__DIR__,"/benchmark.jl"]))
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
    enable_action_pw = false,
    check_repeat_obs = false
)
pft_planner = solve(pft_solver, LightDark)

pomcpow_solver = POMCPOWSolver(
    max_time=t,
    tree_queries = 10_000_000,
    max_depth = d,
    k_observation = 5.0,
    alpha_observation = 1/15,
    criterion = MaxUCB(90.0),
    tree_in_info=false,
    enable_action_pw = false,
    check_repeat_obs = false
)
pomcpow_planner = solve(pomcpow_solver, LightDark)

N = 500
r_pft, r_pomcp = benchmark(
    LightDark,
    BootstrapFilter(LightDark, 1_000),
    [pft_planner, pomcpow_planner],
    N,
    d
)

Distributed.rmprocs(worker_ids)

histogram([r_pft r_pomcp], alpha=0.5, labels=["PFT-DPW" "POMCPOW"], bins=10, normalize=true, legend=:topleft)
title!("LightDark1D Benchmark\nt=$(t)s, d=$d, N=$N")
xlabel!("Returns")
ylabel!("Density")
mean(r_pft)
mean(r_pomcp)
std(r_pft)/sqrt(N)
std(r_pomcp)/sqrt(N)
