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

struct RandomSolver
    rng::AbstractRNG
end

RandomSolver() = RandomSolver(Random.default_rng())

struct RandomPolicy{A,RNG<:AbstractRNG}
    actions::A
    rng::RNG
end

POMDPs.solve(s::RandomSolver, pomdp::POMDP) = RandomPolicy(actions(pomdp), s.rng)

POMDPs.action(s::RandomPolicy, ::Any) = rand(s.rng, s.actions)

struct FastRandomSolver
    rng::Random.AbstractRNG
    d::Union{Nothing, Int}
end

FastRandomSolver(d=nothing) = FastRandomSolver(Random.default_rng(), d)

struct FastRandomRolloutEstimator{ObsRequired, A, RNG <:AbstractRNG}
    actions::A
    rng::RNG
    d::Union{Nothing, Int}
end

function FastRandomRolloutEstimator(pomdp::POMDP, estim::FastRandomSolver, obs_req::Bool)
    RNG = typeof(estim.rng)
    act = actions(pomdp)
    A = typeof(act)
    return FastRandomRolloutEstimator{obs_req,A,RNG}(act, estim.rng, estim.d)
end

const FRRE = FastRandomRolloutEstimator{false, A, RNG} where {A,RNG}
const ObsReqFRRE = FastRandomRolloutEstimator{true, A, RNG} where {A,RNG}

POMDPs.action(p::FastRandomRolloutEstimator, ::Any) = rand(p.rng, p.actions)

function MCTS.convert_estimator(estimator::FastRandomSolver, ::Any, pomdp::POMDP)
    return FastRandomRolloutEstimator(pomdp, estimator, is_obs_required(pomdp))
end

function sr_gen(estimator::FastRandomRolloutEstimator{false}, pomdp::POMDP{S,A}, s::S, a::A) where {S,A}
    return sr_gen(Val(false), estimator.rng, pomdp, s, a)
end

function sr_gen(estimator::FastRandomRolloutEstimator{true}, pomdp::POMDP{S,A}, s::S, a::A) where {S,A}
    return sr_gen(Val(true), estimator.rng, pomdp, s, a)
end

function sr_gen(::Val{false}, rng::AbstractRNG, pomdp::POMDP{S,A}, s::S, a::A) where {S,A}
    sp = rand(rng, transition(pomdp, s, a))
    r = reward(pomdp, s, a, sp)
    return sp, r
end

function sr_gen(::Val{true}, rng::AbstractRNG, pomdp::POMDP{S,A}, s::S, a::A) where {S,A}
    return @gen(:sp,:r)(pomdp, s, a, rng)
end

function MCTS.estimate_value(estimator::FastRandomRolloutEstimator, pomdp::POMDP{S}, s::S, max_depth::Int) where S

    disc = 1.0
    r_total = 0.0
    rng = estimator.rng
    step = 1

    while !isterminal(pomdp, s) && step ≤ max_depth

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
    max_depth = isnothing(est.d) ? d : est.d
    for (s,w) in weighted_particles(b)
        v += w*MCTS.estimate_value(est, pomdp, s, max_depth::Int)
    end
    return v
end

## POROLLOUT

"""Automatically replaced with PFTFilter"""
struct PlaceHolderUpdater <: Updater end

POMDPs.update(::PlaceHolderUpdater, args...) = error("Updater is placeholder")

struct PORollout
    solver::Solver
    updater::Updater
    rng::AbstractRNG
    n_rollouts::Int # number of rollouts per value estimation. if 0, rollout all particles.
    d::Union{Nothing, Int}
end

function PORollout(sol::Solver, d=10; n_rollouts::Int=1, rng::AbstractRNG=Random.default_rng())
    return PORollout(sol, PlaceHolderUpdater(), rng, n_rollouts, d)
end

struct SolvedPORollout{P<:Policy,U<:Updater,RNG<:AbstractRNG,PMEM<:ParticleCollection}
    policy::P
    updater::U
    rng::RNG
    n_rollouts::Int
    ib::PMEM
    rb::PMEM
    d::Union{Nothing, Int}
end

function MCTS.convert_estimator(est::ParticleFilterTrees.PORollout, sol, pomdp::POMDP)
    upd = est.updater
    if upd isa PlaceHolderUpdater
        upd = PFTFilter(pomdp, sol.n_particles)
    end
    S = statetype(pomdp)
    policy = MCTS.convert_to_policy(est.solver, pomdp)
    ParticleFilterTrees.SolvedPORollout(
        policy,
        upd,
        est.rng,
        est.n_rollouts,
        ParticleCollection(Vector{S}(undef,sol.n_particles)),
        ParticleCollection(Vector{S}(undef,sol.n_particles)),
        est.d
    )
end

function MCTS.estimate_value(est::BasicPOMCP.SolvedFOValue, pomdp::POMDP{S}, s::S, d::Int) where S
    POMDPs.value(est.policy, s)
end

function MCTS.estimate_value(est::ParticleFilterTrees.SolvedPORollout, pomdp::POMDP{S}, b::PFTBelief{S}, d::Int) where S
    b_ = initialize_belief!(est.updater, b, est.ib)
    if est.n_rollouts < 1
        return full_rollout(est, pomdp, b_, d)
    else
        return partial_rollout(est, pomdp, b_, d)
    end
end

function full_rollout(est::ParticleFilterTrees.SolvedPORollout, pomdp::POMDP{S}, b::ParticleCollection{S}, d::Int) where S
    v = 0.0
    b_ = est.rb
    for (s,w) in weighted_particles(b)
        b_.particles .= est.ib.particles
        v += w*rollout(est, pomdp, b_, s, d)
    end
    return v
end

function partial_rollout(est::ParticleFilterTrees.SolvedPORollout, pomdp::POMDP{S}, b::ParticleCollection{S}, d::Int) where S
    v = 0.0
    max_depth = isnothing(est.d) ? d : est.d
    b_ = est.rb
    for _ in 1:est.n_rollouts
        b_.particles .= est.ib.particles
        s = rand(est.rng, b)
        v += rollout(est, pomdp, b_, s, max_depth::Int)
    end
    return v/est.n_rollouts
end

function rollout(est::ParticleFilterTrees.SolvedPORollout, pomdp::POMDP{S}, b::ParticleCollection{S}, s::S, max_depth::Int) where S
    updater = est.updater
    rng = est.rng
    policy = est.policy

    disc = 1.0
    r_total = 0.0
    step = 1

    while !isterminal(pomdp, s) && step ≤ max_depth

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
