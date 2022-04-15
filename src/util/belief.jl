struct PFTBelief{T} <: AbstractParticleBelief{T}
    particles::Vector{T}
    weights::Vector{Float64}
    non_terminal_ws::Float64
end

function PFTBelief(particles::Vector{T}, weights::Vector{Float64}, pomdp::POMDP) where {T}
    terminal_ws = 0.0
    for (s,w) in zip(particles, weights)
        !isterminal(pomdp, s) && (terminal_ws += w)
    end
    return PFTBelief{T}(particles, weights, terminal_ws)
end

@inline ParticleFilters.n_particles(b::PFTBelief) = length(b.particles)
@inline ParticleFilters.particles(p::PFTBelief) = p.particles
ParticleFilters.weighted_particles(b::PFTBelief) = (
    b.particles[i]=>b.weights[i] for i in 1:length(b.particles)
)
@inline ParticleFilters.weight_sum(b::PFTBelief) = 1.0
@inline ParticleFilters.weight(b::PFTBelief, i::Int) = b.weights[i]
@inline ParticleFilters.particle(b::PFTBelief, i::Int) = b.particles[i]
@inline ParticleFilters.weights(b::PFTBelief) = b.weights

function Random.rand(rng::AbstractRNG, b::PFTBelief)
    t = rand(rng)
    i = 1
    cw = b.weights[1]
    while cw < t && i < length(b.weights)
        i += 1
        @inbounds cw += weight(b,i)
    end
    return particle(b,i)
end

function non_terminal_sample(rng::AbstractRNG, pomdp::POMDP, b::PFTBelief)
    t = rand(rng)*b.non_terminal_ws
    i = 1
    cw = isterminal(pomdp,particle(b,1)) ? 0.0 : weight(b,1)
    while cw < t && i < length(b.weights)
        i += 1
        isterminal(pomdp,particle(b,i)) && continue
        @inbounds cw += weight(b,i)
    end
    return i
end

Random.rand(b::PFTBelief) = Random.rand(Random.GLOBAL_RNG, b)
