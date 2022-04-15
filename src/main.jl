function POMDPs.solve(sol::PFTDPWSolver, pomdp::POMDP{S,A,O}) where {S,A,O}
    act = actions(pomdp)
    a = rand(act)

    solved_ve = convert_estimator(sol.value_estimator, sol, pomdp)
    obs_req = is_obs_required(pomdp)
    if !sol.enable_action_pw
        try
            SA = length(act)
            @assert SA < Inf
        catch e
            error("Action space should have some defined length if enable_action_pw=false")
        end
    else
        SA = -1
    end

    cache = BeliefCache{S}(sol)
    sz = min(sol.tree_queries, sol.treecache_size)
    return PFTDPWPlanner(
        pomdp,
        sol,
        PFTDPWTree{S,A,O}(sz, sol.check_repeat_obs, sol.k_o, sol.k_a),
        solved_ve,
        a,
        Val(obs_req),
        cache
    )
end

""" WIP → FIX"""
function POMDPs.solve(sol::SparsePFTSolver, pomdp::POMDP{S,A,O}) where {S,A,O}
    act = actions(pomdp)
    a = rand(act)

    solved_action_selector = solve(sol.action_selector, pomdp)
    obs_req = is_obs_required(pomdp)
    if !sol.enable_action_pw
        try
            SA = length(act)
            @assert SA < Inf
        catch e
            error("Action space should have some defined length if enable_action_pw=false")
        end
    else
        SA = -1
    end

    cache = BeliefCache{S}(sol)
    sz = min(sol.tree_queries, sol.treecache_size)
    return SparsePFTPlanner(
        pomdp,
        sol,
        PFTDPWTree{S,A,O}(sz, sol.check_repeat_obs, sol.k_o, sol.k_a),
        solved_action_selector,
        a,
        Val(obs_req),
        cache
    )
end

function POMDPModelTools.action_info(planner::AbstractPFTPlanner, b)
    t0 = time()

    sol = planner.sol
    pomdp = planner.pomdp
    max_iter = sol.tree_queries
    max_time = sol.max_time
    max_depth = sol.max_depth

    A = actiontype(pomdp)

    empty!(planner.tree)
    free!(planner.cache)
    insert_root!(planner, b)

    iter = 0
    if planner.sol.check_repeat_obs
        while (time()-t0 < max_time) && (iter < max_iter)
            obs_check_search(planner, 1, max_depth)
            iter += 1
        end
    else
        while (time()-t0 < max_time) && (iter < max_iter)
            no_obs_check_search(planner, 1, max_depth)
            iter += 1
        end
    end

    a, a_idx = UCB1action(planner, planner.tree, 1, 0.0)

    # If not enough time for even 1 tree query -> give random action
    iszero(a_idx) && ( a = rand(sol.rng, actions(pomdp)) )

    return a::A, (
        n_iter = iter::Int,
        tree = planner.tree::PFTDPWTree,
        time = (time() - t0)::Float64
        )
end

function POMDPs.action(planner::AbstractPFTPlanner, b)
    return first(action_info(planner, b))
end

function isterminalbelief(pomdp::POMDP, b::PFTBelief)
    !any(!isterminal(pomdp,s)*w>0 for (s,w) in weighted_particles(b))
end

function Base.empty!(tree::PFTDPWTree)
    empty!(tree.Nh)
    empty!(tree.Nha)
    empty!(tree.Qha)

    empty!(tree.b)
    empty!(tree.b_children)
    empty!(tree.b_rewards)

    empty!(tree.bao_children)
    empty!(tree.ba_children)

    nothing
end
