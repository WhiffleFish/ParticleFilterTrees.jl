function GenBelief(
    planner::PFTDPWPlanner,
    pomdp::POMDP{S,A,O},
    b::B,
    a::A
    )::Tuple{PFTBelief, O, Float64} where {S,A,O,B<:PFTBelief{S}}

    p_idx = non_terminal_sample(rng, pomdp, b)

    sample_s = particle(b, p_idx)

    sample_sp, sample_obs, sample_r = @gen(:sp,:o,:r)(pomdp, sample_s, a, rng)

    return GenBelief(planner, pomdp, b, a, sample_obs, p_idx, sample_sp, sample_r)
end

function GenBelief(
    planner::PFTDPWPlanner,
    pomdp::POMDP{S,A,O},
    b::RegPFTBelief{S},
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

    for (s,w) in zip(bp_particles, bp_weights)
        !isterminal(pomdp, s) && (bp_terminal_ws += w)
    end

    # Just return belief of all terminal states so it's never ever ever touched or looked at again
    if iszero(bp_terminal_ws)
        # Oh god I hate it
        terminal_particle = sample_s
        for s in bp_particles
            if isterminal(pomdp, s)
                terminal_particle = s
                break
            end
        end
        return RegPFTBelief(fill(terminal_particle, N), fill(inv(N), N), 0.0)::PFTBelief{S}, sample_obs::O, 0.0
    end

    bp = RegPFTBelief(bp_particles, bp_weights, bp_terminal_ws)

    return bp::RegPFTBelief{S}, weighted_return::Float64
end

function GenBelief(
    planner::PFTDPWPlanner,
    pomdp::POMDP{S,A,O},
    b::ResamplingPFTBelief{S},
    a::A,
    o::O,
    p_idx::Int,
    sample_sp::S,
    sample_r::Float64
    )::Tuple{PFTBelief{S}, Float64} where {S,A,O}

    rng = planner.sol.rng
    planner._weight_cache.sum = 0.0
    N = n_particles(b)
    weighted_return = 0.0

    bp_particles = Vector{S}(undef, N)
    bp_terminal_ws = 0.0

    w = b.p_weight
    for (i,s) in enumerate(particles(b))
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
            planner._particle_cache[i] = sp
            w_i = pdf(POMDPs.observation(pomdp, s, a, sp), o)
            planner._weight_cache[i] = w_i
            planner._weight_cache.sum += w_i
        end

        weighted_return += r*w
    end

    resample!(rng, planner._particle_cache, planner._weight_cache, bp_particles)

    bp = ResamplingPFTBelief(bp_particles, pomdp)

    return bp::ResamplingPFTBelief{S}, weighted_return::Float64
end

function incremental_avg(Qhat::Float64, Q::Float64, N::Int)
    return Qhat + (Q - Qhat)/N
end

function resample!(rng::AbstractRNG, cache::Vector{S}, w::StatsBase.Weights{Float64,Float64,Vector{Float64}}, bp_particles::Vector{S}) where {S}
    StatsBase.alias_sample!(rng, cache, w, bp_particles)
    nothing
end
