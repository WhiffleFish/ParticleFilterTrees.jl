function GenBelief(rng::AbstractRNG, pomdp::POMDP{S,A,O}, b::WeightedParticleBelief{S}, a)::Tuple{WeightedParticleBelief, O, Float64} where {S,A,O}

    sample_obs = @gen(:o)(pomdp, rand(b), a, rng)

    bp = WeightedParticleBelief(sizehint!(S[],n_particles(b)), sizehint!(Float64[],n_particles(b)))
    weighted_return = 0.0
    for (s,w) in weighted_particles(b)
        (sp, r) = @gen(:sp,:r)(pomdp, s, a, rng)
        push!(bp.particles, sp)

        push!(bp.weights, w*pdf(POMDPs.observation(pomdp, s, a, sp), sample_obs))
        weighted_return += r*w/b.weight_sum
    end
    bp.weight_sum = sum(bp.weights)
    bp.weights ./= bp.weight_sum
    bp.weight_sum = 1.0
    return bp, sample_obs, weighted_return
end

function incremental_avg(Qhat::Float64, Q::Float64, N::Int)
    return Qhat + (Q - Qhat)/N
end
