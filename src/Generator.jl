function GenBelief(pomdp::POMDP, b::WeightedParticleBelief{S}, a)::Tuple{WeightedParticleBelief, Float64} where S
    # NOTE: Need POMDP, rng inputs for gen function

    # Gen rand obs for particle re-weighting
    sample_obs = @gen(:o)(pomdp, rand(b), a, rng)

    new_states = S[]
    new_weights = Float64[]
    weighted_return = 0.0
    for (s,w) in weighted_particles(b)
        (sp, o, r, _) = POMDPs.gen(pomdp, s, a, rng)
        push!(new_states, sp)

        push!(new_weights, pdf(POMDPs.observation(pomdp, s, a, sp), sample_obs))
        weighted_return += r*w # NOTE: Calculate return based on particle weight?
    end
    bp = WeightedParticleBelief(new_states, new_weights)
    return bp, o, weighted_return
end

# NOTE: PFT-DPW does not seem to take observation weighting into account, so
# how do we reweight the particles after propagation?
