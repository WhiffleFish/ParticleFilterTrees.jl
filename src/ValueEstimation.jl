#=
Value Estimators must have the following implemented
- convert_estimator(unsolved_estimator, solver, problem)
- estimate_value(solved_estimator,  mdp, belief, depth)
=#

function is_obs_required(pomdp::POMDP)
    s,sp = rand(initialstate(pomdp)), rand(initialstate(pomdp))
    a = rand(actions(pomdp))

    obs_req = false
    try # very crude. would like to use `applicable` or `hasmethod` but these fail with quickpomdps
        POMDPs.reward(pomdp, s, a, sp)
    catch e
        obs_req = true
    end

    return obs_req
end

struct FastRandomSolver{RNG <: Random.AbstractRNG}
    rng::RNG
end

FastRandomSolver() = FastRandomSolver(Xorshifts.Xoroshiro128Star())

struct FastRandomRolloutEstimator{ObsRequired, A, RNG <:AbstractRNG}
    actions::A
    rng::RNG
end

function FastRandomRolloutEstimator(pomdp::POMDP, estim::FastRandomSolver, obs_req::Bool)
    RNG = typeof(estim.rng)
    act = actions(pomdp)
    A = typeof(act)
    return FastRandomRolloutEstimator{obs_req,A,RNG}(act, estim.rng)
end

const FRRE = FastRandomRolloutEstimator{false, A, RNG} where {A,RNG}
const ObsReqFRRE = FastRandomRolloutEstimator{true, A, RNG} where {A,RNG}

action(p::FastRandomRolloutEstimator, ::Any) = rand(p.rng, p.actions)

function convert_estimator(estimator::FastRandomSolver, ::Any, pomdp::POMDP)

    obs_req = is_obs_required(pomdp)

    return FastRandomRolloutEstimator(pomdp, estimator, obs_req)
end

function estimate_value(estimator::FastRandomRolloutEstimator, pomdp::POMDP, b::PFTBelief, depth::Int)
    r = 0.0
    for (s,w) in weighted_particles(b)
        r += w*rollout(estimator, pomdp, s, depth)
    end
    return r::Float64
end

function sr_gen(estimator::FastRandomRolloutEstimator{false}, pomdp::POMDP{S,A,O}, s::S, a::A) where {S,A,O}
    sp = rand(estimator.rng, transition(pomdp, s, a))
    r = reward(pomdp, s, a, sp)
    return sp, r
end

function sr_gen(estimator::FastRandomRolloutEstimator{true}, pomdp::POMDP{S,A,O}, s::S, a::A) where {S,A,O}
    return @gen(:sp,:r)(pomdp, s, a, estimator.rng)
end

function rollout(estimator::FastRandomRolloutEstimator, pomdp::POMDP{S,A,O}, s::S, depth::Int) where {S,A,O}

    disc = 1.0
    r_total = 0.0
    rng = estimator.rng
    step = 1

    while !isterminal(pomdp, s) && step <= depth

        a = action(estimator, s)

        sp,r = sr_gen(estimator, pomdp, s, a)

        r_total += disc*r

        s = sp

        disc *= discount(pomdp)
        step += 1
    end

    return r_total
end

# Fallback for non-belief defined value estimators
function estimate_value(est, pomdp::POMDP, b::PFTBelief, d::Int)
    v = 0.0
    for (s,w) in weighted_particles(b)
        v += w*estimate_value(est, pomdp, s, d) # estim def in terms of state
    end
    return v
end

function estimate_value(est::BasicPOMCP.SolvedFOValue, pomdp::POMDP{S}, s::S, d::Int) where S
    POMDPs.value(est.policy, s)
end
