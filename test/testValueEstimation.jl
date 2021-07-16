using PFTDPW
using Random
using RandomNumbers: Xorshifts
using POMDPSimulators
using BeliefUpdaters
using Statistics
using POMDPModels
using POMDPs
include(joinpath(@__DIR__,"LightDarkPOMDP.jl"))


RNG_type = typeof(Xorshifts.Xoroshiro128Star())

struct RandomRollout{A, RNG <: AbstractRNG} <: Policy
    rng::RNG
    actions::A
end

RandomRollout(pomdp::POMDP) = RandomRollout(Xorshifts.Xoroshiro128Star(), actions(pomdp))

POMDPs.action(p::RandomRollout, ::Any) = rand(p.rng, p.actions)

function rollout(pomdp::POMDP, p::RandomRollout, b::PFTBelief, d::Int)::Float64
    r = 0.0
    sim = RolloutSimulator(rng = p.rng, max_steps = d)
    upd = NothingUpdater()
    for (s,w) in PFTDPW.weighted_particles(b)
        r_s = simulate(
                sim,
                pomdp,
                p,
                upd,
                b,
                s
            )::Float64
        r += w*r_s # weight sum assumed to be 1.0
    end
    return r::Float64
end

tiger = TigerPOMDP()
baby = BabyPOMDP()

function test_norm_estim(pomdp::POMDP, d::Int, N::Int)
    ib = PFTDPW.initial_belief(pomdp, initialstate(pomdp), 1_000)
    rand_policy = RandomRollout(pomdp)
    return [rollout(pomdp, rand_policy, ib, d) for _ in 1:N]
end

function test_fast_estim(pomdp::POMDP, d::Int, N::Int)
    ib = PFTDPW.initial_belief(pomdp, initialstate(pomdp), 1_000)
    s,sp = rand(initialstate(pomdp)),rand(initialstate(pomdp))
    act = actions(pomdp)
    a = rand(act)
    rng = Xorshifts.Xoroshiro128Star()

    obs_req = false
    try
        reward(pomdp, s, a, sp)
    catch
        obs_rew = true
    end
    FFRE = PFTDPW.FastRandomRolloutEstimator{obs_req, typeof(act), typeof(rng)}(
        actions(pomdp),
        rng
    )

    return [PFTDPW.estimate_value(FFRE, pomdp, ib, d) for _ in 1:N]

end


pomdp = LightDark
a = test_norm_estim(pomdp, 10, 1000)
b = test_fast_estim(pomdp, 10, 1000)
mean(a)
mean(b)
