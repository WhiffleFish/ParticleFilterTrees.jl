function POMDPs.solve(sol::PFTDPWSolver, pomdp::POMDP{S,A,O}) where {S,A,O}
    act = actions(pomdp)

    solved_ve = MCTS.convert_estimator(sol.value_estimator, sol, pomdp)
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
        Val(obs_req),
        cache
    )
end

@inline function search_method(@nospecialize sol::PFTDPWSolver)
    return if sol.check_repeat_obs
        if sol.vanilla
            vanilla_obs_check_search
        else
            obs_check_search
        end
    else
        if sol.vanilla
            vanilla_no_obs_check_search
        else
            no_obs_check_search
        end
    end
end

"""
function barrier for all possible search functions
"""
function _search(f, planner, t0)
    sol = planner.sol
    max_iter = sol.tree_queries
    max_time = sol.max_time
    max_depth = sol.max_depth
    iter = 0
    while (time()-t0 < max_time) && (iter < max_iter)
        f(planner, 1, max_depth)
        iter += 1
    end
    return iter
end

function POMDPTools.action_info(planner::PFTDPWPlanner, b)
    t0 = time()

    sol = planner.sol
    pomdp = planner.pomdp

    A = actiontype(pomdp)

    empty!(planner.tree)
    free!(planner.cache)
    insert_root!(planner, b)

    iter = _search(search_method(sol), planner, t0)

    a = if isempty(first(planner.tree.b_children))
        sol.default_action(pomdp, b)
    else
        first(select_best(MaxQ(), planner.tree, 1))
    end

    return a, (
        n_iter = iter,
        tree = planner.tree,
        time = time() - t0
        )
end

function POMDPs.action(planner::PFTDPWPlanner, b)
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
