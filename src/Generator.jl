function GenBelief(rng::AbstractRNG, pomdp::POMDP{S,A,O}, b::WeightedParticleBelief{S}, a)::Tuple{WeightedParticleBelief, O, Float64} where {S,A,O}

    N = n_particles(b)
    weighted_return = 0.0

    bp = WeightedParticleBelief(sizehint!(S[],N), sizehint!(Float64[],N))

    p_idx = StatsBase.sample(StatsBase.weights(b.weights))
    sample_obs = nothing

    sample_s = particle(b, p_idx)
    if isterminal(pomdp, sample_s)
        w = inv(N)
        for _ in 1:N; push!(bp.particles,sample_s); push!(bp.weights, w); end
        sample_obs = first(Vector{O}(undef,1)) # Random Observation
        weighted_return = 0.0
        return bp::WeightedParticleBelief{S}, sample_obs::O, weighted_return::Float64
    end

    # Propagation
    for (i,(s,w)) in enumerate(weighted_particles(b))
        if i == p_idx
            (sp, sample_obs, r) = @gen(:sp,:o,:r)(pomdp, s, a, rng)
        else
            if !isterminal(pomdp, s)
                (sp, r) = @gen(:sp,:r)(pomdp, s, a, rng)
            else
                (sp,r) = (s, 0.0)
            end
        end

        push!(bp.particles, sp)

        weighted_return += r*w
    end

    # Reweighting
    @inbounds for i in 1:N
        s = particle(b, i)
        sp = particle(bp, i)
        w = weight(b, i)
        push!(bp.weights, w*pdf(POMDPs.observation(pomdp, s, a, sp), sample_obs))
    end

    if !iszero(sum(bp.weights))
        normalize!(bp.weights, 1)
    else
        bp.weights .= 1/N
    end

    bp.weight_sum = 1.0

    return bp::WeightedParticleBelief{S}, sample_obs::O, weighted_return::Float64
end

function incremental_avg(Qhat::Float64, Q::Float64, N::Int)
    return Qhat + (Q - Qhat)/N
end
