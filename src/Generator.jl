function GenBelief(
    planner::PFTDPWPlanner,
    pomdp::POMDP{S,A,O},
    b::B,
    a::A
    ) where {S,A,O,B<:PFTBelief{S}}

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
        terminal_particle = sample_sp
        for s in bp_particles
            if isterminal(pomdp, s)
                terminal_particle = s
                break
            end
        end
        return RegPFTBelief(fill(terminal_particle, N), fill(inv(N), N), 0.0)::PFTBelief{S}, o::O, weighted_return::Float64
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

    cache = planner._RWCache
    rng = planner.sol.rng
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
            cache.particle_cache[i] = sp
            w_i = pdf(POMDPs.observation(pomdp, s, a, sp), o)
            cache.weight_cache[i] = w_i
        end

        weighted_return += r*w
    end

    rw_alias_sample!(rng, cache, bp_particles)

    bp = ResamplingPFTBelief(bp_particles, pomdp)

    return bp::ResamplingPFTBelief{S}, o::O, weighted_return::Float64
end

function rw_make_alias_table!(w::AbstractVector{Float64}, wsum::Float64,
                           a::AbstractVector{Float64},
                           alias::AbstractVector{Int}, cache::ReweightCache)

    n = length(w)
    length(a) == length(alias) == n ||
        throw(DimensionMismatch("Inconsistent array lengths."))

    ac = n / wsum
    for i = 1:n
        @inbounds a[i] = w[i] * ac
    end

    larges = cache.larges
    smalls = cache.smalls
    kl = 0  # actual number of larges
    ks = 0  # actual number of smalls

    for i = 1:n
        @inbounds ai = a[i]
        if ai > 1.0
            larges[kl+=1] = i  # push to larges
        elseif ai < 1.0
            smalls[ks+=1] = i  # push to smalls
        end
    end

    while kl > 0 && ks > 0
        s = smalls[ks]; ks -= 1  # pop from smalls
        l = larges[kl]; kl -= 1  # pop from larges
        @inbounds alias[s] = l
        @inbounds al = a[l] = (a[l] - 1.0) + a[s]
        if al > 1.0
            larges[kl+=1] = l  # push to larges
        else
            smalls[ks+=1] = l  # push to smalls
        end
    end

    # this loop should be redundant, except for rounding
    for i = 1:ks
        @inbounds a[smalls[i]] = 1.0
    end
    nothing
end

# https://github.com/JuliaStats/StatsBase.jl/blob/9f1d7aafa86f8771a995b54de1e2432c6e9f55a0/src/sampling.jl#L513-L525
function rw_alias_sample!(rng::AbstractRNG, cache::ReweightCache, x::AbstractArray)
    a = cache.particle_cache
    wv = cache.weight_cache
    n = length(a)
    length(wv) == n || throw(DimensionMismatch("Inconsistent lengths."))

    # create alias table
    ap = cache.ap
    alias = cache.alias
    rw_make_alias_table!(wv, sum(wv), ap, alias, cache)

    # sampling
    s = 1:n
    for i = 1:length(x)
        j = rand(rng, s)
        x[i] = rand(rng) < ap[j] ? a[j] : a[alias[j]]
    end
    nothing
end

function incremental_avg(Qhat::Float64, Q::Float64, N::Int)
    return Qhat + (Q - Qhat)/N
end
