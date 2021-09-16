function GenBelief(
    planner::PFTDPWPlanner,
    pomdp::POMDP{S,A,O},
    b::PFTBelief{S},
    a::A
    )::Tuple{PFTBelief, O, Float64} where {S,A,O}

    rng = planner.sol.rng
    N = n_particles(b)
    weighted_return = 0.0

    p_idx = non_terminal_sample(rng, pomdp, b)

    sample_s = particle(b, p_idx)
    sample_sp, sample_obs, sample_r = @gen(:sp,:o,:r)(pomdp, sample_s, a, rng)

    return GenBelief(
        planner,
        pomdp,
        b,
        a,
        sample_obs,
        p_idx,
        sample_sp,
        sample_r
    )
end

function GenBelief(
    planner::PFTDPWPlanner,
    pomdp::POMDP{S,A,O},
    b::PFTBelief{S},
    a::A,
    o::O,
    p_idx::Int,
    sample_sp::S,
    sample_r::Float64
    ) where {S,A,O}

    rng = planner.sol.rng
    N = n_particles(b)
    weighted_return = 0.0

    bp_particles, bp_weights = gen_empty_belief(planner.cache, N)

    for (i,(s,w)) in enumerate(weighted_particles(b))
        # Propagation
        if i === p_idx
            (sp, r) = sample_sp, sample_r
        else
            if !isterminal(pomdp, s)
                (sp, r) = sr_gen(planner.obs_req, rng, pomdp, s, a) # @gen(:sp,:r)(pomdp, s, a, rng)
            else
                (sp,r) = (s, 0.0)
            end
        end

        # Reweighting
        @inbounds begin
            bp_particles[i] = sp
            w = weight(b, i)
            bp_weights[i] = w*pdf(POMDPs.observation(pomdp, s, a, sp), o)
        end

        weighted_return += r*w
    end

    if !all(iszero, bp_weights)
        normalize!(bp_weights, 1)
    else
        fill!(bp_weights, inv(N))
    end

    bp = PFTBelief(bp_particles, bp_weights, pomdp)

    return bp::PFTBelief{S}, o::O, weighted_return::Float64
end

function incremental_avg(Qhat::Float64, Q::Float64, N::Int)
    return Qhat + (Q - Qhat)/N
end
