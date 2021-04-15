"""
Return weighted average rollout
"""
function rollout(solver::Solver, b::WeightedParticleBelief)::Float64 # Paralellizable
    r = 0.0
    for (s,w) in weighted_particles(b)
        r += w*simulate(
            solver.value_estimator,
            solver.pomdp,
            solver.default_policy,
            init_state=s
            )
    end
    return r
end


function simulate(pomdp::POMDP, tree::PFTDPWTree, sol::PFTDPWSolver, b_idx::Int, d::Int)::Float64
    if d == 0
        return 0.0
    end
    a = action_prog_widen(pomdp, tree, sol, b_idx)
    ba_idx = tree.b_children[b_idx][a]
    if length(tree.ba_children[ba_idx]) <= sol.k_o*tree.Nha[ba_idx]^sol.alpha_o
        # NOTE: MAKE SURE WE'RE NOT OVERWRITING BELIEFS
        bp, o, r = GenBelief(pomdp, tree.b[b_idx], a)
        insert_belief!(tree, bp, ba_idx, o, r)
        total = r + discount(pomdp)*rollout(sol, bp, d-1)
    else
        o, bp_idx = rand(tree.ba_children[ba_idx])
        r = tree.b_rewards[bp_idx]
        total = r + discount(pomdp)*simulate(pomdp, tree, sol, bp_idx, d-1)
    end
    tree.Nh[b_idx] += 1
    tree.Nha[ba_idx] += 1
    tree.Qha[ba_idx] = tree.Qha[ba_idx] + (total - tree.Qha[ba_idx])/tree.Nha[ba_idx]
end
