function next_action(pomdp::POMDP)
    return rand(actions(pomdp)) # TODO: include user-provided RNG
end

function UCB(Q::Float64, Nh::Int, Nha::Int, c::Float64)::Float64
    return Nha > 0 ? Q + c*sqrt(log(Nh)/Nha) : Inf
end

function UCB1action(planner::PFTDPWPlanner, tree::PFTDPWTree{S,A,O}, b_idx::Int, c::Float64) where {S,A,O}

    max_ucb = -Inf
    opt_a = planner._placeholder_a
    opt_idx = 0
    for (a,ba_idx) in tree.b_children[b_idx]
        ucb = UCB(tree.Qha[ba_idx], tree.Nh[b_idx], tree.Nha[ba_idx], c)
        if ucb > max_ucb
            max_ucb = ucb
            opt_a = a
            opt_idx = ba_idx
        end
    end
    return opt_a::A, opt_idx::Int
end

function act_prog_widen(planner::PFTDPWPlanner, b_idx::Int)
    sol = planner.sol
    tree = planner.tree

    k_a, alpha_a, c = sol.k_a, sol.alpha_a, sol.c

    if length(tree.b_children[b_idx]) <= k_a*tree.Nh[b_idx]^alpha_a
        a = next_action(planner.pomdp)
        if isempty(filter(x->x[1] == a, tree.b_children[b_idx]))
            insert_action!(tree, b_idx, a)
        end
    end

    return UCB1action(planner, tree, b_idx, c)
end
