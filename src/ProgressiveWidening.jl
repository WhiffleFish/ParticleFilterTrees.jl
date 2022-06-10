function next_action(rng::AbstractRNG, pomdp::POMDP)
    return rand(rng, actions(pomdp))
end

function UCB1action(planner::PFTDPWPlanner, tree::PFTDPWTree{S,A}, b_idx::Int, c::Float64) where {S,A}

    lnh = log(tree.Nh[b_idx])
    local opt_a::A
    max_ucb = -Inf
    opt_idx = 0

    for (a,ba_idx) in tree.b_children[b_idx]
        Nha = tree.Nha[ba_idx]
        iszero(Nha) && return a::A, ba_idx::Int
        Q̂ = tree.Qha[ba_idx]
        ucb = Q̂ + c*sqrt(lnh / Nha)

        if ucb > max_ucb
            max_ucb = ucb
            opt_a = a
            opt_idx = ba_idx
        end
    end
    return opt_a, opt_idx
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

    return UCB1action(planner, tree, b_idx, c)
end

function act_widen(planner::PFTDPWPlanner, b_idx::Int)
    sol = planner.sol
    tree = planner.tree

    if isempty(tree.b_children[b_idx])
        for a in actions(planner.pomdp)
            insert_action!(planner, tree, b_idx, a)
        end
    end

    return UCB1action(planner, tree, b_idx, sol.c)
end
