function GenBelief(
    planner::PFTDPWPlanner,
    pomdp::POMDP{S,A,O},
    b::WeightedParticleBelief{S},
    a::A
    )::Tuple{WeightedParticleBelief, O, Float64} where {S,A,O}

    rng = planner.sol.rng
    N = n_particles(b)
    weighted_return = 0.0

    bp = WeightedParticleBelief(Vector{S}(undef, N), Vector{Float64}(undef, N))

    p_idx = StatsBase.sample(StatsBase.weights(b.weights))
    sample_obs = planner._placeholder_o

    sample_s = particle(b, p_idx)
    if isterminal(pomdp, sample_s)
        w = inv(N)
        fill!(bp.particles,sample_s)
        fill!(bp.weights,w)
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

        @inbounds bp.particles[i] = sp

        weighted_return += r*w
    end

    # Reweighting
    @inbounds for i in 1:N
        s = particle(b, i)
        sp = particle(bp, i)
        w = weight(b, i)
        bp.weights[i] = w*pdf(POMDPs.observation(pomdp, s, a, sp), sample_obs)
    end

    if !iszero(sum(bp.weights))
        normalize!(bp.weights, 1)
    else
        fill!(bp.weights, inv(N))
    end

    bp.weight_sum = 1.0

    return bp::WeightedParticleBelief{S}, sample_obs::O, weighted_return::Float64
end

function incremental_avg(Qhat::Float64, Q::Float64, N::Int)
    return Qhat + (Q - Qhat)/N
end
