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

function BasicPOMCP.convert_estimator(estimator::FastRandomSolver, ::Any, pomdp::POMDP)

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

function estimate_value(estimator::FastRandomRolloutEstimator, pomdp::POMDP{S,A,O}, s::S, depth::Int) where {S,A,O}

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
function estimate_value(est, pomdp::POMDP{S}, b::PFTBelief{S}, d::Int) where S
    v = 0.0
    for (s,w) in weighted_particles(b)
        v += w*estimate_value(est, pomdp, s, d) # estim def in terms of state
    end
    return v
end

function estimate_value(est::BasicPOMCP.SolvedFOValue, pomdp::POMDP{S}, s::S, d::Int) where S
    POMDPs.value(est.policy, s)
end


function rollout(est::BasicPOMCP.SolvedPORollout, pomdp::POMDP{S}, b::ParticleCollection{S}, s::S, d::Int) where S

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

struct PFTFilter{PM,RNG<:AbstractRNG,PMEM} <: Updater
    pomdp::PM
    rng::RNG
    _particle_memory::PMEM
    _weight_memory::Vector{Float64}
end

function PFTFilter(pomdp::POMDP, n_p::Int)
    S = statetype(pomdp)
    return PFTFilter(
        pomdp,
        Xorshifts.Xoroshiro128Star(),
        ParticleCollection(Vector{S}(undef,n_p)),
        Vector{Float64}(undef, n_p)
        )
end

### BAD: For predict! source and destination arrays point to the same place / they are the same
function initialize_belief(pf::PFTFilter, b::PFTBelief)
    b_ = pf._particle_memory
    resample!(b,b_,pf.rng)
end

function initialize_belief(up::BasicParticleFilter, b::PFTBelief)
    return ParticleCollection([rand(b) for _ in 1:up.n_init])
end

function estimate_value(est::BasicPOMCP.SolvedPORollout, pomdp::POMDP{S}, b::PFTBelief{S}, d::Int) where S
    v = 0.0
    b_ = initialize_belief(est.updater, b)
    for (s,w) in weighted_particles(b)
        v += w*rollout(est, pomdp, b_, s, d)
    end
    return v
end

function update!(up::PFTFilter, b::ParticleCollection, a, o)
    pm = up._particle_memory
    wm = up._weight_memory
    predict!(pm.particles, up.pomdp, b, a, up.rng)
    reweight!(wm, up.pomdp, b, a, pm.particles, o)
    resample!(pm, wm, b, up.rng)
end

function resample!(b::ParticleCollection, w::Vector{Float64}, bp::ParticleCollection, rng::AbstractRNG)
    n_p = length(b.particles)
    ws = sum(w)
    ps = bp.particles

    r = rand(rng)*ws/n_p
    c = w[1]
    i = 1
    U = r
    for m in 1:n_p
        while U > c && i < n_p
            i += 1
            c += w[i]
        end
        U += ws/n_p
        ps[m] = b.particles[i]
    end
    return bp
end

function resample!(b::PFTBelief, bp::ParticleCollection, rng::AbstractRNG)
    n_p = n_particles(b)
    w = b.weights
    ws = sum(w)
    ps = bp.particles

    r = rand(rng)*ws/n_p
    c = w[1]
    i = 1
    U = r
    for m in 1:n_p
        while U > c && i < n_p
            i += 1
            c += w[i]
        end
        U += ws/n_p
        ps[m] = b.particles[i]
    end
    return bp
end


# function rollout(est::BasicPOMCP.SolvedPORollout, pomdp::POMDP{S}, b::ParticleCollection{S}, s::S, d::Int) where S
#
#     updater = est.updater
#     rng = est.rng
#     policy = est.policy
#     disc = 1.0
#     r_total = 0.0
#     step = 1
#
#     while !isterminal(pomdp, s) && step <= d
#
#         a = ParticleFilters.action(policy, b)
#
#         sp, o, r = @gen(:sp,:o,:r)(pomdp, s, a, rng)
#
#         r_total += disc*r
#
#         s = sp
#
#         update!(updater, b, a, o)
#
#         disc *= discount(pomdp)
#         step += 1
#     end
#
#     return r_total
# end
