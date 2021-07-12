mutable struct BeliefCache{S}
    particles::Vector{Vector{S}}
    weights::Vector{Vector{Float64}}
    count::Int
    max_size::Int
end

function BeliefCache{S}(sol::PFTDPWSolver) where S
    sz = min(sol.tree_queries, 100_000)
    n_p = sol.n_particles
    return BeliefCache{S}(
        [Vector{S}(undef, n_p) for _ in  1:sz],
        [Vector{Float64}(undef, n_p) for _ in 1:sz],
        0,
        sz
    )
end


free!(cache::BeliefCache) = (cache.count = 0)
