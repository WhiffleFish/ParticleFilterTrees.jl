abstract type PFTBelief{T} end

struct ResamplingPFTBelief{T} <: PFTBelief{T}
    particles::Vector{T}
    p_weight::Float64
    non_terminal_ws::Float64
end

ResamplingPFTBelief(p::Vector{T}, ntw::Float64) where {T} = ResamplingPFTBelief{T}(p, inv(length(p)), ntw)

function ResamplingPFTBelief(p::Vector{S}, pomdp::POMDP{S,A,O}) where {S,A,O}
    bp_terminal_ws = 0.0
    w = inv(length(p))
    for s in p
        !isterminal(pomdp, s) && (bp_terminal_ws += w)
    end
    return ResamplingPFTBelief{S}(p, w, bp_terminal_ws)
end

@inline n_particles(b::ResamplingPFTBelief) = length(b.particles)
@inline particles(p::ResamplingPFTBelief) = p.particles
weighted_particles(b::ResamplingPFTBelief) = (b.particles[i]=>b.p_weight for i in 1:n_particles(b))
@inline weight_sum(b::ResamplingPFTBelief) = 1.0
@inline weight(b::ResamplingPFTBelief, i::Int) = b.p_weight
@inline particle(b::ResamplingPFTBelief, i::Int) = b.particles[i]
@inline weights(b::ResamplingPFTBelief) = fill(b.p_weight,n_particles(b))

function Random.rand(rng::AbstractRNG, b::ResamplingPFTBelief)
    i = rand(rng, 1:n_particles(b))
    return particle(b,i)
end

function non_terminal_sample(rng::AbstractRNG, pomdp::POMDP, b::ResamplingPFTBelief)
    t = rand(rng)*b.non_terminal_ws
    i = 1
    N = n_particles(b)
    w = inv(N)
    cw = isterminal(pomdp, particle(b,1)) ? 0.0 : w
    while cw < t && i < N
        i += 1
        isterminal(pomdp,particle(b,i)) && continue
        # @inbounds cw += w
        cw += w
    end
    return i
end

Random.rand(b::ResamplingPFTBelief) = Random.rand(Random.GLOBAL_RNG, b)


struct RegPFTBelief{T} <: PFTBelief{T}
    particles::Vector{T}
    weights::Vector{Float64}
    non_terminal_ws::Float64
end

@inline n_particles(b::RegPFTBelief) = length(b.particles)
@inline particles(p::RegPFTBelief) = p.particles
weighted_particles(b::RegPFTBelief) = (b.particles[i]=>b.weights[i] for i in 1:length(b.particles))
@inline weight_sum(b::RegPFTBelief) = 1.0
@inline weight(b::RegPFTBelief, i::Int) = b.weights[i]
@inline particle(b::RegPFTBelief, i::Int) = b.particles[i]
@inline weights(b::RegPFTBelief) = b.weights

function Random.rand(rng::AbstractRNG, b::RegPFTBelief)
    t = rand(rng)
    i = 1
    cw = b.weights[1]
    while cw < t && i < length(b.weights)
        i += 1
        # @inbounds cw += weight(b,i)
        cw += weight(b,i)
    end
    return particle(b,i)
end

function non_terminal_sample(rng::AbstractRNG, pomdp::POMDP, b::RegPFTBelief)
    t = rand(rng)*b.non_terminal_ws
    i = 1
    cw = isterminal(pomdp, particle(b,1)) ? 0.0 : weight(b,1)
    while cw < t && i < length(b.weights)
        i += 1
        isterminal(pomdp,particle(b,i)) && continue
        # @inbounds cw += weight(b,i)
        cw += weight(b,i)
    end
    return i
end

Random.rand(b::RegPFTBelief) = Random.rand(Random.GLOBAL_RNG, b)

StatsBase.mean(b::RegPFTBelief{T}) where {T <: Number} = dot(b.weights, b.particles)
StatsBase.mean(b::RegPFTBelief{T}) where {T <: Vector} = reduce(hcat, b.particles) * b.weights
function StatsBase.cov(b::RegPFTBelief{T}) where {T <: Number} # uncorrected covariance
    centralized = b.particles .- mean(b)
    sum(centralized .* b.weights .* centralized)
end
function StatsBase.cov(b::RegPFTBelief{T}) where {T <: Vector} # uncorrected covariance
    centralized = reduce(hcat, b.particles) .- mean(b)
    (centralized .* b.weights') * centralized'
end
