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

    bp_particles = Vector{S}(undef, N)
    bp_weights = Vector{Float64}(undef, N)
    bp_terminal_ws = 0.0

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
            bp_particles[i] = sp
            w = weight(b, i)
            bp_w = w*pdf(POMDPs.observation(pomdp, s, a, sp), sample_obs)
            bp_weights[i] = bp_w
        end
        weighted_return += r*w
    end

    if !iszero(sum(bp_weights))
        normalize!(bp_weights, 1)
    else
        fill!(bp_weights, inv(N))
    end

    bp = PFTBelief(bp_particles, bp_weights, pomdp)

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

    bp_particles = Vector{S}(undef, N)
    bp_weights = Vector{Float64}(undef, N)
    bp_terminal_ws = 0.0

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
            bp_particles[i] = sp
            w = weight(b, i)
            bp_weights[i] = w*pdf(POMDPs.observation(pomdp, s, a, sp), o)
        end

        weighted_return += r*w
    end

    if !iszero(sum(bp_weights))
        normalize!(bp_weights, 1)
    else
        fill!(bp_weights, inv(N))
    end

    bp = PFTBelief(bp_particles, bp_weights, pomdp)

    return bp::PFTBelief{S}, weighted_return::Float64
end

function incremental_avg(Qhat::Float64, Q::Float64, N::Int)
    return Qhat + (Q - Qhat)/N
end
