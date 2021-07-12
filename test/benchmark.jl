using POMDPs
using POMDPSimulators
using Random

function benchmark(pomdp::POMDP, upd::Updater, planners::Vector{Policy}, N::Int, max_steps::Int)
    rewards = Vector{Float64}[]

    for planner in planners

        sims = [POMDPSimulators.Sim(
            pomdp,
            planner,
            upd,
            max_steps = max_steps,
            simulator=RolloutSimulator(max_steps=max_steps, rng=MersenneTwister(rand(UInt32)))
        ) for _ in 1:N]

        res = run_parallel(sims, show_progress=true)

        push!(rewards, res.reward)
    end

    return rewards
end

function benchmark(pomdp::POMDP, upd::Updater, planner::Policy, N::Int, max_steps::Int)

    sims = [POMDPSimulators.Sim(
        pomdp,
        planner,
        upd,
        max_steps = max_steps,
        simulator=RolloutSimulator(max_steps=max_steps, rng=MersenneTwister(rand(UInt32)))
    ) for _ in 1:N]

    res = run_parallel(sims, show_progress=true, proc_warn=false)

    return res.reward
end
