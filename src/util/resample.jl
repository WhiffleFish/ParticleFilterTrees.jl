function resample!(planner::PFTDPWPlanner, b::PFTBelief)
    n_p = n_particles(b)
    ps = planner.cache.resample
    w = weights(b)
    ws = 1.0
    r = rand(planner.sol.rng)/n_p
    c = w[1]
    i = 1
    U = r

    for m in 1:n_p
        while U > c && i < n_p
            i += 1
            c += w[i]
        end
        U += ws/n_p
        @inbounds ps[m] = b.particles[i]
    end

    copyto!(b.particles, ps)
    fill!(b.weights, inv(n_p))
    return b
end
