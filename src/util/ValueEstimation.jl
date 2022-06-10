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

struct RandomSolver{RNG<:AbstractRNG}
    rng::RNG
end

RandomSolver() = RandomSolver(Random.default_rng())

struct RandomPolicy{A,RNG<:AbstractRNG}
    actions::A
    rng::RNG
end

POMDPs.solve(s::RandomSolver, pomdp::POMDP) = RandomPolicy(actions(pomdp), s.rng)

POMDPs.action(s::RandomPolicy, ::Any) = rand(s.rng, s.actions)

struct FastRandomSolver{RNG <: Random.AbstractRNG}
    rng::RNG
end

FastRandomSolver() = FastRandomSolver(Random.default_rng())

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

POMDPs.action(p::FastRandomRolloutEstimator, ::Any) = rand(p.rng, p.actions)

function MCTS.convert_estimator(estimator::FastRandomSolver, ::Any, pomdp::POMDP)

    obs_req = is_obs_required(pomdp)

    return FastRandomRolloutEstimator(pomdp, estimator, obs_req)
end

function sr_gen(estimator::FastRandomRolloutEstimator{false}, pomdp::POMDP{S,A,O}, s::S, a::A) where {S,A,O}
    return sr_gen(Val(false), estimator.rng, pomdp, s, a)
end

function sr_gen(estimator::FastRandomRolloutEstimator{true}, pomdp::POMDP{S,A,O}, s::S, a::A) where {S,A,O}
    return sr_gen(Val(true), estimator.rng, pomdp, s, a)
end

function sr_gen(::Val{false}, rng::AbstractRNG, pomdp::POMDP{S,A,O}, s::S, a::A) where {S,A,O}
    sp = rand(rng, transition(pomdp, s, a))
    r = reward(pomdp, s, a, sp)
    return sp, r
end

function sr_gen(::Val{true}, rng::AbstractRNG, pomdp::POMDP{S,A,O}, s::S, a::A) where {S,A,O}
    return @gen(:sp,:r)(pomdp, s, a, rng)
end

function MCTS.estimate_value(estimator::FastRandomRolloutEstimator, pomdp::POMDP{S,A,O}, s::S, depth::Int) where {S,A,O}

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
function MCTS.estimate_value(est, pomdp::POMDP{S}, b::PFTBelief{S}, d::Int) where S
    v = 0.0
    for (s,w) in weighted_particles(b)
        v += w*MCTS.estimate_value(est, pomdp, s, d) # estim def in terms of state
    end
    return v
end

## POROLLOUT

"""Automatically replaced with PFTFilter"""
struct PlaceHolderUpdater <: Updater end

POMDPs.update(::PlaceHolderUpdater, args...) = error("Updater is placeholder")

struct PORollout{SOL<:Solver, UPD<:Updater, RNG<:AbstractRNG}
    solver::SOL
    updater::UPD
    rng::RNG
    n_rollouts::Int # number of rollouts per value estimation. if 0, rollout all particles.
end

function PORollout(sol::Solver, rng::AbstractRNG; n_rollouts::Int=1)
    return PORollout(sol, PlaceHolderUpdater(), rng, n_rollouts)
end

function PORollout(sol::Solver; n_rollouts::Int=1)
    return PORollout(sol, Random.default_rng(), n_rollouts=n_rollouts)
end

struct SolvedPORollout{P<:Policy,U<:Updater,RNG<:AbstractRNG,PMEM<:ParticleCollection}
    policy::P
    updater::U
    rng::RNG
    n_rollouts::Int
    ib::PMEM
    rb::PMEM
end

function MCTS.convert_estimator(est::PFTDPW.PORollout, sol, pomdp::POMDP)
    upd = est.updater
    if upd isa PlaceHolderUpdater
        upd = PFTFilter(pomdp, sol.n_particles)
    end
    S = statetype(pomdp)
    policy = MCTS.convert_to_policy(est.solver, pomdp)
    PFTDPW.SolvedPORollout(
        policy,
        upd,
        est.rng,
        est.n_rollouts,
        ParticleCollection(Vector{S}(undef,sol.n_particles)),
        ParticleCollection(Vector{S}(undef,sol.n_particles))
    )
end

function MCTS.estimate_value(est::BasicPOMCP.SolvedFOValue, pomdp::POMDP{S}, s::S, d::Int) where S
    POMDPs.value(est.policy, s)
end

function MCTS.estimate_value(est::PFTDPW.SolvedPORollout, pomdp::POMDP{S}, b::PFTBelief{S}, d::Int) where S
    b_ = initialize_belief!(est.updater, b, est.ib)
    if est.n_rollouts < 1
        return full_rollout(est, pomdp, b_, d)
    else
        return partial_rollout(est, pomdp, b_, d)
    end
end

function full_rollout(est::PFTDPW.SolvedPORollout, pomdp::POMDP{S}, b::ParticleCollection{S}, d::Int) where S
    v = 0.0
    b_ = est.rb
    for (s,w) in weighted_particles(b)
        b_.particles .= est.ib.particles
        v += w*rollout(est, pomdp, b_, s, d)
    end
    return v
end

function partial_rollout(est::PFTDPW.SolvedPORollout, pomdp::POMDP{S}, b::ParticleCollection{S}, d::Int) where S
    v = 0.0
    w = 1/est.n_rollouts
    b_ = est.rb
    for _ in 1:est.n_rollouts
        b_.particles .= est.ib.particles
        s = rand(est.rng, b)
        v += w*rollout(est, pomdp, b_, s, d)
    end
    return v
end

function rollout(est::PFTDPW.SolvedPORollout, pomdp::POMDP{S}, b::ParticleCollection{S}, s::S, d::Int) where S
    updater = est.updater

    rng = est.rng
    policy = est.policy

    disc = 1.0
    r_total = 0.0
    step = 1

    while !isterminal(pomdp, s) && step <= d

        a = ParticleFilters.action(policy, b)

        sp, o, r = @gen(:sp,:o,:r)(pomdp, s, a, rng)

        r_total += disc*r

        s = sp

        bp = update(updater, b, a, o)
        b = bp

        disc *= discount(pomdp)
        step += 1
    end

    return r_total
end
