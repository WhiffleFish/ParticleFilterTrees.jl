function GenBelief(
    planner::PFTDPWPlanner,
    pomdp::POMDP{S,A,O},
    b::PFTBelief{S},
    a::A
    ) where {S,A,O}

    p_idx = non_terminal_sample(planner.sol.rng, pomdp, b)

    sample_s = particle(b, p_idx)

    sample_sp, sample_obs, sample_r = @gen(:sp,:o,:r)(pomdp, sample_s, a, planner.sol.rng)

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
    ) where {S,A,O}

    rng = planner.sol.rng
    N = n_particles(b)
    weighted_return = 0.0

    bp_particles, bp_weights = get_cached_belief(planner.cache)
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

    bp = RegPFTBelief(bp_particles, bp_weights, bp_terminal_ws)

    return bp::RegPFTBelief{S}, o::O, weighted_return::Float64
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
    ) where {S,A,O}

    cache = planner.cache
    rng = planner.sol.rng
    N = n_particles(b)
    weighted_return = 0.0

    bp_particles = get_cached_particles(cache)
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
            cache.resample_particles[i] = sp
            w_i = pdf(POMDPs.observation(pomdp, s, a, sp), o)
            cache.resample_weights[i] = w_i
        end

        weighted_return += r*w
    end

    resample!(rng, cache, bp_particles)

    bp = ResamplingPFTBelief(bp_particles, pomdp)

    return bp::ResamplingPFTBelief{S}, o::O, weighted_return::Float64
end


# Low variance resampling from https://github.com/JuliaPOMDP/ParticleFilters.jl/blob/master/src/resamplers.jl
function resample!(rng::AbstractRNG, cache::Cache{S}, ps::Vector{S}) where {S}
    ws = sum(cache.resample_weights)
    N = length(cache.resample_weights)
    r = rand(rng)*ws/N
    c = first(cache.resample_weights)
    i = 1
    U = r
    for m in 1:N
        while U > c && i < N
            i += 1
            c += cache.resample_weights[i]
        end
        U += ws/N
        @inbounds ps[m] = cache.resample_particles[i]
    end
    nothing
end

function incremental_avg(Qhat::Float64, Q::Float64, N::Int)
    return Qhat + (Q - Qhat)/N
end
