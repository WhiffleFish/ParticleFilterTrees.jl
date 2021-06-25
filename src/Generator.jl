function GenBelief(
    planner::PFTDPWPlanner,
    pomdp::POMDP{S,A,O},
    b::PFTBelief{S},
    a::A
    )::Tuple{PFTBelief, O, Float64} where {S,A,O}

    rng = planner.sol.rng
    N = n_particles(b)
    weighted_return = 0.0

    aug_weights = StatsBase.weights(Float64[w*!isterminal(pomdp,s) for (s,w) in weighted_particles(b)])
    p_idx = StatsBase.sample(aug_weights)

    sample_s = particle(b, p_idx)

    sample_sp, sample_obs, sample_r = @gen(:sp,:o,:r)(pomdp, sample_s, a, rng)

    bp = PFTBelief(Vector{S}(undef, N), Vector{Float64}(undef, N))

    for (i,(s,w)) in enumerate(weighted_particles(b))
        # Propagation
        if i == p_idx
            (sp, r) = sample_sp, sample_r
        else
            if !isterminal(pomdp, s)
                (sp, r) = @gen(:sp,:r)(pomdp, s, a, rng)
            else
                (sp,r) = (s, 0.0)
            end
        end

        # Reweighting
        @inbounds begin
            bp.particles[i] = sp
            w = weight(b, i)
            bp.weights[i] = w*pdf(POMDPs.observation(pomdp, s, a, sp), sample_obs)
        end

        weighted_return += r*w
    end

    if !iszero(sum(bp.weights))
        normalize!(bp.weights, 1)
    else
        fill!(bp.weights, inv(N))
    end

    return bp::PFTBelief{S}, sample_obs::O, weighted_return::Float64
end

# Too many args
function ObsCheckGenBelief(
    planner::PFTDPWPlanner,
    pomdp::POMDP{S,A,O},
    b::PFTBelief{S},
    a::A,
    o::O,
    p_idx::Int,
    sample_sp::S,
    sample_r::Float64
    )::Tuple{PFTBelief{S}, Float64} where {S,A,O}

    rng = planner.sol.rng
    N = n_particles(b)
    weighted_return = 0.0

    bp = PFTBelief(Vector{S}(undef, N), Vector{Float64}(undef, N))

    for (i,(s,w)) in enumerate(weighted_particles(b))
        # Propagation
        if i == p_idx
            (sp, r) = sample_sp, sample_r
        else
            if !isterminal(pomdp, s)
                (sp, r) = @gen(:sp,:r)(pomdp, s, a, rng)
            else
                (sp,r) = (s, 0.0)
            end
        end

        # Reweighting
        @inbounds begin
            bp.particles[i] = sp
            w = weight(b, i)
            bp.weights[i] = w*pdf(POMDPs.observation(pomdp, s, a, sp), o)
        end

        weighted_return += r*w
    end

    if !iszero(sum(bp.weights))
        normalize!(bp.weights, 1)
    else
        fill!(bp.weights, inv(N))
    end

    return bp::PFTBelief{S}, weighted_return::Float64
end

function incremental_avg(Qhat::Float64, Q::Float64, N::Int)
    return Qhat + (Q - Qhat)/N
end
