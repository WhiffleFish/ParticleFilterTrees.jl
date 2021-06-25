struct PFTBelief{T}
    particles::Vector{T}
    weights::Vector{Float64}
end

n_particles(b::PFTBelief) = length(b.particles)
particles(p::PFTBelief) = p.particles
weighted_particles(b::PFTBelief) = (b.particles[i]=>b.weights[i] for i in 1:length(b.particles))
weight_sum(b::PFTBelief) = 1.0
weight(b::PFTBelief, i::Int) = b.weights[i]
particle(b::PFTBelief, i::Int) = b.particles[i]
weights(b::PFTBelief) = b.weights

function Random.rand(rng::AbstractRNG, b::PFTBelief)
    t = rand(rng)
    i = 1
    cw = b.weights[1]
    while cw < t && i < length(b.weights)
        i += 1
        @inbounds cw += b.weights[i]
    end
    return particles(b)[i]
end

Random.rand(b::PFTBelief) = Random.rand(Random.GLOBAL_RNG, b)

StatsBase.mean(b::PFTBelief{T}) where {T <: Number} = dot(b.weights, b.particles)
StatsBase.mean(b::PFTBelief{T}) where {T <: Vector} = reduce(hcat, b.particles) * b.weights
function StatsBase.cov(b::PFTBelief{T}) where {T <: Number} # uncorrected covariance
    centralized = b.particles .- mean(b)
    sum(centralized .* b.weights .* centralized)
end
function StatsBase.cov(b::PFTBelief{T}) where {T <: Vector} # uncorrected covariance
    centralized = reduce(hcat, b.particles) .- mean(b)
    (centralized .* b.weights') * centralized'
end
