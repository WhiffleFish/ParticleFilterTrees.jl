struct Cache{S}
    resample_particles::Vector{S}
    resample_weights::Vector{Float64}

    particles::Vector{Vector{S}}
    weights::Vector{Vector{Float64}}

    capacity::Int
    n_particles::Int
    n::Ref{Int}
end

used(cache::Cache) = cache.n[]
increment!(cache::Cache) = cache.n[] += 1
free!(cache::Cache) = cache.n[] = 0

function get_cached_belief(cache::Cache{S}) where {S}
    n = used(cache)
    if n < cache.capacity
        increment!(cache)
        return cache.particles[n+1], cache.weights[n+1]
    else
        return Vector{S}(undef, cache.n_particles), Vector{Float64}(undef, cache.n_particles)
    end
end

function get_cached_particles(cache::Cache{S}) where {S}
    n = used(cache)
    if n < cache.capacity
        increment!(cache)
        return cache.particles[n+1]
    else
        return Vector{S}(undef,cache.n_particles)
    end
end
