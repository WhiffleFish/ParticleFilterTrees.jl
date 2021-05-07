function rollout(planner::Policy, solver::Solver, b::WeightedParticleBelief, d::Int)::Float64 # Paralellizable
    r = 0.0
    sim = RolloutSimulator(rng = solver.rng, max_steps = d)
    for (s,w) in weighted_particles(b)
        r_sim = simulate(
                    sim::RolloutSimulator,
                    planner.pomdp::TigerPOMDP,
                    planner.rollout_policy::RandomRollout,
                    planner.updater::NothingUpdater,
                    b::WeightedParticleBelief,
                    s::Bool
                )::Float64
        r += (w/b.weight_sum)*r_sim
    end
    return r::Float64
end


b0 = initial_belief(initialstate(tiger),100)
sim = RolloutSimulator(rng = Random.GLOBAL_RNG, max_steps = 10)
s = rand(b0)

# @profiler [simulate(sim, tiger, planner.rollout_policy, planner.updater, b0, s) for _ in 1:1_000_000]

@profiler [rollout(planner, solver, b0, 10) for _ in 1:10000]

@benchmark rollout(planner, solver, b0, 10)

@benchmark rollout(planner, solver, b0, 10)

@benchmark simulate(sim, tiger, planner.rollout_policy, planner.updater, b0, s)
