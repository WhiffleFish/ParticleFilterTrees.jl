function no_obs_check_search(planner::PFTDPWPlanner, b_idx::Int, d::Int)::Float64
    tree = planner.tree
    pomdp = planner.pomdp
    sol = planner.sol

    if iszero(d) || tree.b[b_idx].non_terminal_ws < 1e-300
        return 0.0
    end

    a, ba_idx = act_prog_widen(planner, b_idx)
    if length(tree.ba_children[ba_idx]) <= sol.k_o*tree.Nha[ba_idx]^sol.alpha_o
        bp, o, r = GenBelief(planner, pomdp, tree.b[b_idx], a)
        insert_belief!(tree, bp, ba_idx, o, r, planner)
        ro = estimate_value(planner.solved_VE, pomdp, bp, d-1)
        total = r + discount(pomdp)*ro
    else
        bp_idx = rand(sol.rng, tree.ba_children[ba_idx])
        r = tree.b_rewards[bp_idx]
        total = r + discount(pomdp)*no_obs_check_search(planner, bp_idx, d-1)
    end

    tree.Nh[b_idx] += 1
    tree.Nha[ba_idx] += 1
    tree.Qha[ba_idx] = incremental_avg(tree.Qha[ba_idx], total, tree.Nha[ba_idx])

    return total::Float64
end


function obs_check_search(planner::PFTDPWPlanner, b_idx::Int, d::Int)::Float64
    tree = planner.tree
    pomdp = planner.pomdp
    sol = planner.sol

    if iszero(d) || tree.b[b_idx].non_terminal_ws < 1e-300
        return 0.0
    end

    a, ba_idx = act_prog_widen(planner, b_idx)
    if length(tree.ba_children[ba_idx]) <= sol.k_o*tree.Nha[ba_idx]^sol.alpha_o

        b = tree.b[b_idx]
        p_idx = non_terminal_sample(sol.rng, pomdp, b)
        sample_s = particle(b, p_idx)
        sample_sp, o, sample_r = @gen(:sp,:o,:r)(pomdp, sample_s, a)

        if !haskey(tree.bao_children, (ba_idx, o))
            bp, _, r = GenBelief(planner, pomdp, b, a, o, p_idx, sample_sp, sample_r)
            insert_belief!(tree, bp, ba_idx, o, r, planner)
            ro = estimate_value(planner.solved_VE, pomdp, bp, d-1)
            total = r + discount(pomdp)*ro

        else
            bp_idx::Int = tree.bao_children[(ba_idx,o)]
            push!(tree.ba_children[ba_idx], bp_idx)
            r = tree.b_rewards[bp_idx]
            total = r + discount(pomdp)*obs_check_search(planner, bp_idx, d-1)
        end
    else
        bp_idx = rand(sol.rng, tree.ba_children[ba_idx])
        r = tree.b_rewards[bp_idx::Int]
        total = r + discount(pomdp)*obs_check_search(planner, bp_idx, d-1)

    end

    tree.Nh[b_idx] += 1
    tree.Nha[ba_idx] += 1
    tree.Qha[ba_idx] = incremental_avg(tree.Qha[ba_idx], total, tree.Nha[ba_idx])

    return total::Float64
end
