"""
Return weighted average rollout
"""
function rollout(planner::PFTDPWPlanner, b::WeightedParticleBelief, d::Int)::Float64
    r = 0.0
    sim = RolloutSimulator(rng = planner.sol.rng, max_steps = d)
    for (s,w) in weighted_particles(b)
        r_s = simulate(
                sim,
                planner.pomdp,
                planner.rollout_policy,
                planner.sol.updater,
                b,
                s
            )::Float64
        r += w*r_s # weight sum assumed to be 1.0
    end
    return r::Float64
end


function search(planner::PFTDPWPlanner, b_idx::Int, d::Int)::Float64
    tree = planner.tree
    pomdp = planner.pomdp
    sol = planner.sol

    if iszero(d) || isterminalbelief(pomdp, tree.b[b_idx])
        return 0.0
    end

    a, ba_idx = act_prog_widen(pomdp, tree, sol, b_idx)
    if length(tree.ba_children[ba_idx]) <= sol.k_o*tree.Nha[ba_idx]^sol.alpha_o
        bp, o, r = GenBelief(sol.rng, pomdp, tree.b[b_idx], a)

        if !haskey(tree.bao_children, (ba_idx,o))
            insert_belief!(tree, bp, ba_idx, o, r, planner)
            ro = rollout(planner, bp, d-1)
            total = r + discount(pomdp)*ro
        else
            bp_idx = tree.bao_children[(ba_idx,o)]
            r = tree.b_rewards[bp_idx]
            total = r + discount(pomdp)*search(planner, bp_idx, d-1)
        end
    else
        bp_idx = rand(tree.ba_children[ba_idx])
        r = tree.b_rewards[bp_idx]
        total = r + discount(pomdp)*search(planner, bp_idx, d-1)
    end
    tree.Nh[b_idx] += 1
    tree.Nha[ba_idx] += 1

    tree.Qha[ba_idx] = incremental_avg(tree.Qha[ba_idx], total, tree.Nha[ba_idx])

    return total::Float64
end

function POMDPs.solve(sol::PFTDPWSolver, pomdp::POMDP{S,A,O})::PFTDPWPlanner where {S,A,O}
    return PFTDPWPlanner(pomdp, sol, PFTDPWTree{S,A,O}(1))
end

function POMDPModelTools.action_info(planner::PFTDPWPlanner, b)
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
        search(planner, 1, max_depth)
        iter += 1
    end

    a, a_idx = UCB1action(planner.tree, 1, 0.0)
    if a_idx == 0; a = rand(actions(pomdp)); end

    return a::A, Dict{Symbol, Any}(
        :action => a::A,
        :n_iter => iter::Int,
        :tree => planner.tree::PFTDPWTree,
        :time => (time() - t0)::Float64
        )
end

function POMDPs.action(planner::PFTDPWPlanner, b)
    return first(action_info(planner, b))
end

function isterminalbelief(pomdp::POMDP, b::WeightedParticleBelief)
    all(isterminal(pomdp, s) for s in particles(b))
end
