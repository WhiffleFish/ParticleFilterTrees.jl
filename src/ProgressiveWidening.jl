# TODO: Remove assertion statements

# NOTE: Maybe use more sophisticated sampling strategy
function next_action(pomdp::POMDP)
    return rand(actions(pomdp)) # TODO: include user-provided RNG
end

function UCB(Q::Float64, Nh::Int, Nha::Int, c::Float64)::Float64
    return Q + c*sqrt(log(Nh)/Nha)
end

# NOTE: May want to also return action index
function UCB1action(tree::PFTDPWTree, b_idx::Int)
    max_ucb = -Inf
    opt_a = nothing
    for (a,a_idx) in tree.b_children[b_idx]
        ucb = UCB(Qha[a_idx], Nh[b_idx], Nha[a_idx])
        if ucb > max_ucb
            max_ucb = ucb
            opt_a = a
        end
    end
    return opt_a
end

function act_prog_widen(pomdp::POMDP, tree::PFTDPWTree, sol::PFTDPWSolver, b_idx::Int)

    k_a, alpha_a, c = sol.k_a, sol.alpha_a, sol.c

    if tree.Nh[b_idx] <= k_a*Nh^alpha_a
        a = next_action(pomdp)
        if !haskey(tree.b_children[b_idx], a)
            insert_action!(tree, b_idx, a)
        end
    end
    return ICB1action(tree, b_idx)
end
