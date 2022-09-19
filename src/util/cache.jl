mutable struct BeliefCache{S}
    particles::Vector{Vector{S}}
    weights::Vector{Vector{Float64}}
    resample::Vector{S}
    count::Int
    capacity::Int
end

function BeliefCache{S}(sol::PFTDPWSolver) where S
    sz = min(sol.tree_queries, sol.beliefcache_size)
    n_p = sol.n_particles
    return BeliefCache{S}(
        [Vector{S}(undef, n_p) for _ in  1:sz],
        [Vector{Float64}(undef, n_p) for _ in 1:sz],
        Vector{S}(undef, n_p),
        0,
        sz
    )
end

function gen_empty_belief(cache::BeliefCache{S}, N::Int) where {S}
    cache.count += 1
    if cache.count ≤ cache.capacity
        return cache.particles[cache.count]::Vector{S}, cache.weights[cache.count]::Vector{Float64}
    else
        return Vector{S}(undef, N), Vector{Float64}(undef, N)
    end
end

free!(cache::BeliefCache) = (cache.count = 0)
