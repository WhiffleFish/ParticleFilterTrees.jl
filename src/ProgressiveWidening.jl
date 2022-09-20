function next_action(rng::AbstractRNG, pomdp::POMDP)
    return rand(rng, actions(pomdp))
end

function act_prog_widen(planner::PFTDPWPlanner, b_idx::Int)
    if planner.sol.enable_action_pw
        return progressive_widen(planner, b_idx)
    else
        return act_widen(planner, b_idx)
    end
end

function progressive_widen(planner::PFTDPWPlanner, b_idx::Int)
    (;tree, sol) = planner
    (;k_a, alpha_a, c) = sol

    if length(tree.b_children[b_idx]) ≤ k_a*tree.Nh[b_idx]^alpha_a
        a = next_action(sol.rng, planner.pomdp)
        if !any(x[1] == a for x in tree.b_children[b_idx])
            insert_action!(planner, tree, b_idx, a)
        end
    end

    return select_best(sol.criterion, tree, b_idx)
end

function act_widen(planner::PFTDPWPlanner, b_idx::Int)
    sol = planner.sol
    tree = planner.tree

    if isempty(tree.b_children[b_idx])
        for a in actions(planner.pomdp)
            insert_action!(planner, tree, b_idx, a)
        end
    end

    return select_best(sol.criterion, tree, b_idx)
end
