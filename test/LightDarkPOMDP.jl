using QuickPOMDPs
using Distributions
using POMDPTools

const R = 60
const LIGHT_LOC = 10

LightDarkPOMDP() = QuickPOMDP(
    states = -R:R+1, # r+1 is a terminal state
    stateindex = s -> s + R + 1,
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

    reward = function (s, a)
        if iszero(a)
            return iszero(s) ? 100.0 : -100.0
        else
            return -1.0
        end
    end,

    initialstate = POMDPTools.Uniform(div(-R::Int,2):div(R::Int,2))
)
