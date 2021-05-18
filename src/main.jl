"""
Return weighted average rollout
"""
function rollout(planner::Policy, solver::Solver, b::WeightedParticleBelief, d::Int)::Float64 # Paralellizable
    r = 0.0
    sim = RolloutSimulator(rng = solver.rng, max_steps = d)
    for (s,w) in weighted_particles(b)
        r_s = simulate(
                sim,
                planner.pomdp,
                planner.rollout_policy,
                planner.sol.updater,
                b,
                s
            )::Float64
        r += (w/b.weight_sum)*r_s
    end
    return r::Float64
end


function search(planner::Policy, sol::PFTDPWSolver, b_idx::Int, d::Int)::Float64
    tree = planner.tree
    pomdp = planner.pomdp

    if iszero(d)
        return 0.0
    end

    a, ba_idx = act_prog_widen(pomdp, tree, sol, b_idx)
    if length(tree.ba_children[ba_idx]) <= sol.k_o*tree.Nha[ba_idx]^sol.alpha_o
        bp, o, r = GenBelief(sol.rng, pomdp, tree.b[b_idx], a)

        if !haskey(tree.bao_children, (ba_idx,o))
            insert_belief!(tree, bp, ba_idx, o, r, planner)
            ro = rollout(planner, sol, bp, d-1)
            total = r + discount(pomdp)*ro
        else
            bp_idx = tree.bao_children[(ba_idx,o)]
            r = tree.b_rewards[bp_idx]
            total = r + discount(pomdp)*search(planner, sol, bp_idx, d-1)
        end
    else
        bp_idx = rand(tree.ba_children[ba_idx])
        r = tree.b_rewards[bp_idx]
        total = r + discount(pomdp)*search(planner, sol, bp_idx, d-1)
    end
    tree.Nh[b_idx] += 1
    tree.Nha[ba_idx] += 1

    tree.Qha[ba_idx] = incremental_avg(tree.Qha[ba_idx], total, tree.Nha[ba_idx])

    return total::Float64
end

function POMDPs.solve(pomdp::POMDP{S,A,O}, sol::PFTDPWSolver)::PFTDPWPlanner where {S,A,O}
    return PFTDPWPlanner(pomdp, sol, PFTDPWTree{S,A,O}(1))
end

function POMDPModelTools.action_info(planner::PFTDPWPlanner, b)::Dict{Symbol, Any}
    # NOTE: moved actions to beginning of function for more accurate timing
    t0 = time()

    sol = planner.sol
    pomdp = planner.pomdp
    max_iter = sol.tree_queries
    max_time = sol.max_time
    max_depth = sol.max_depth

    S = statetype(pomdp)
    A = actiontype(pomdp)
    O = obstype(pomdp)

    planner.tree = PFTDPWTree{S,A,O}(sol.tree_queries)
    insert_root!(planner.tree, b, sol.n_particles)

    iter = 0
    while (time()-t0 < max_time) && (iter < max_iter)
        search(planner, sol, 1, max_depth)
        iter += 1
    end

    UCB_a, _ = UCB1action(planner.tree, 1, 0.0)
    a = UCB_a != nothing ? UCB_a : rand(actions(pomdp))

    return Dict{Symbol, Any}(
        :action => a::A,
        :n_iter => iter,
        :tree => planner.tree
        )
end

function POMDPs.action(planner::PFTDPWPlanner, b)
    return action_info(planner, b)[:action]
end
