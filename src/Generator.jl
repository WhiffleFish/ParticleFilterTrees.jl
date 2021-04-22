function GenBelief(rng::AbstractRNG, pomdp::POMDP{S,A,O}, b::WeightedParticleBelief{S}, a)::Tuple{WeightedParticleBelief, O, Float64} where {S,A,O}
    # NOTE: Using Predefined boostrapfilter may be easier/faster
    # -> Bootstrap filter resamples; we don't resample (but maybe?)
    # Gen rand obs for particle re-weighting
    sample_obs = @gen(:o)(pomdp, rand(b), a, rng)

    new_states = S[]
    new_weights = Float64[]
    weighted_return = 0.0
    for (s,w) in weighted_particles(b)
        (sp, r) = @gen(:sp,:r)(pomdp, s, a, rng)
        push!(new_states, sp)

        push!(new_weights, pdf(POMDPs.observation(pomdp, s, a, sp), sample_obs))
        weighted_return += r*w/b.weight_sum
    end
    bp = WeightedParticleBelief(new_states, new_weights./sum(new_weights))
    return bp, sample_obs, weighted_return
end

function incremental_avg(Qhat::Float64, Q::Float64, N::Int)
    return Qhat + (Q - Qhat)/N
end
