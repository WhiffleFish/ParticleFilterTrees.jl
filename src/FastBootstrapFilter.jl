struct PFTFilter{PM<:POMDP,RNG<:AbstractRNG,PMEM} <: Updater
    pomdp::PM
    rng::RNG
    p::PMEM # initial and post-resampling particles (p → pp → p)
    pp::PMEM # post-prediction but pre-resampling particles
    w::Vector{Float64}
end

function PFTFilter(pomdp::POMDP, n_p::Int, rng::AbstractRNG)
    S = statetype(pomdp)
    return PFTFilter(
        pomdp,
        rng,
        ParticleCollection(Vector{S}(undef,n_p)),
        ParticleCollection(Vector{S}(undef,n_p)),
        Vector{Float64}(undef, n_p)
        )
end

PFTFilter(pomdp::POMDP, n_p::Int) = PFTFilter(pomdp, n_p, Xorshifts.Xoroshiro128Star())

function initialize_belief!(pf::PFTFilter, source::PFTBelief, dest::ParticleCollection)
    resample!(source, dest, pf.rng)
end

"""
predict!
    - propagate b(up.p) → up.pp
reweight!
    - update up._particle_weights
    - s ∈ b(up.p), sp ∈ up.pp
resample!
    - resample up.pp → b (up.p)
"""
function update!(up::PFTFilter, b::ParticleCollection, a, o)
    predict!(up.pp, up.pomdp, b, a, up.rng) # b → up.pp
    reweight!(up.w, up.pomdp, b, a, up.pp.particles, o)
    resample!(up.pp, up.w, b, up.rng) # up.pp → b
end

POMDPs.update(up::PFTFilter, b::ParticleCollection, a, o) = update!(up,b,a,o)

function predict!(pm::ParticleCollection, m::POMDP, b::ParticleCollection, a, rng::AbstractRNG)
    all_terminal = true
    pm_particles = pm.particles
    for i in 1:n_particles(b)
        s = particle(b, i)
        if !isterminal(m, s)
            all_terminal = false
            sp = @gen(:sp)(m, s, a, rng)
            @inbounds pm_particles[i] = sp
        end
    end
    return all_terminal
end

function resample!(b::ParticleCollection, w::Vector{Float64}, bp::ParticleCollection, rng::AbstractRNG)
    n_p = n_particles(b)
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
        @inbounds ps[m] = b.particles[i]
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
