using QuickPOMDPs
using Distributions
using BenchmarkTools
using POMDPModelTools
using POMDPs
using PFTDPW
using D3Trees

const R = 60
const LIGHT_LOC = 10

const LightDark = QuickPOMDP(
    states = -R:R+1,                  # r+1 is a terminal state
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

solver = PFTDPWSolver(tree_queries=10_000, k_o=5, k_a=4, max_depth=30, c=100.0, n_particles=100, check_repeat_obs=false, enable_action_pw=false)
planner = solve(solver, LightDark)
@benchmark a_info = action_info(planner, initialstate(LightDark)) (seconds=120)

@benchmark a_info = action_info(planner, initialstate(LightDark)) (seconds=120)

@profiler a_info = action_info(planner, initialstate(LightDark))

a, info = action_info(planner, initialstate(LightDark))
tree= info[:tree]
t = D3Tree(info[:tree])

inchrome(t)
