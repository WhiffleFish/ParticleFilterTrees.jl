using QuickPOMDPs
using Distributions
using BenchmarkTools
using POMDPModelTools
using POMDPs
using POMCPOW
using PFTDPW

const R = 60
const LIGHT_LOC = 10

const pomdp = QuickPOMDP(
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

solver = PFTDPWSolver(tree_queries=10_000, k_o=1, k_a=2, max_depth=20, c=100.0, n_particles=100, check_repeat_obs=false)
planner = solve(solver, pomdp)
@benchmark a_info = action_info(planner, initialstate(pomdp)) (seconds=120)

@benchmark a_info = action_info(planner, initialstate(pomdp)) (seconds=120)

@profiler a_info = action_info(planner, initialstate(pomdp))

a, info = action_info(planner, initialstate(pomdp))
info[:tree]


solver2 = POMCPOWSolver(tree_queries=100_000,check_repeat_obs=false)
planner2 = solve(solver2, pomdp)

@profiler a,info = action_info(planner2,initialstate(pomdp))
