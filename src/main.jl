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


function no_obs_check_search(planner::PFTDPWPlanner, b_idx::Int, d::Int)::Float64
    tree = planner.tree
    pomdp = planner.pomdp
    sol = planner.sol

    if iszero(d) || isterminalbelief(pomdp, tree.b[b_idx])
        return 0.0
    end

    a, ba_idx = act_prog_widen(planner, b_idx)
    if length(tree.ba_children[ba_idx]) <= sol.k_o*tree.Nha[ba_idx]^sol.alpha_o
        bp, o, r = GenBelief(planner, pomdp, tree.b[b_idx], a)

        insert_belief!(tree, bp, ba_idx, o, r, planner)
        ro = rollout(planner, bp, d-1)
        total = r + discount(pomdp)*ro

    else
        bp_idx = rand(tree.ba_children[ba_idx])
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

    if iszero(d) || isterminalbelief(pomdp, tree.b[b_idx])
        return 0.0
    end

    a, ba_idx = act_prog_widen(planner, b_idx)
    if sum(tree.obs_weights[ba_idx]) <= sol.k_o*tree.Nha[ba_idx]^sol.alpha_o
        bp, o, r = GenBelief(planner, pomdp, tree.b[b_idx], a)

        if !haskey(tree.bao_children, (ba_idx,o))
            insert_belief!(tree, bp, ba_idx, o, r, planner)
            ro = rollout(planner, bp, d-1)
            total = r + discount(pomdp)*ro
        else
            @inbounds begin
                bp_idx = tree.bao_children[(ba_idx,o)]

                ow = tree.obs_weights[ba_idx]
                w_loc = findfirst(x->x==bp_idx, tree.ba_children[ba_idx])
                ow[w_loc] += 1

                r = tree.b_rewards[bp_idx]
            end
            total = r + discount(pomdp)*obs_check_search(planner, bp_idx, d-1)
        end
    else

        w = StatsBase.weights(tree.obs_weights[ba_idx])
        bp_idx = tree.ba_children[ba_idx][StatsBase.sample(w)]

        r = tree.b_rewards[bp_idx]
        total = r + discount(pomdp)*obs_check_search(planner, bp_idx, d-1)
    end
    tree.Nh[b_idx] += 1
    tree.Nha[ba_idx] += 1

    tree.Qha[ba_idx] = incremental_avg(tree.Qha[ba_idx], total, tree.Nha[ba_idx])

    return total::Float64
end

function POMDPs.solve(sol::PFTDPWSolver, pomdp::POMDP{S,A,O})::PFTDPWPlanner where {S,A,O}
    a, o = get_placeholders(pomdp)
    if !sol.enable_action_pw
        try
            @assert length(actions(pomdp)) < Inf
        catch e
            error("Action space should have some defined length if enable_action_pw=false")
        end
    end
    return PFTDPWPlanner(pomdp, sol, PFTDPWTree{S,A,O}(sol.tree_queries, sol.check_repeat_obs), RandomRollout(pomdp), a, o)
end

function POMDPModelTools.action_info(planner::PFTDPWPlanner, b)
    t0 = time()

    sol = planner.sol
    pomdp = planner.pomdp
    max_iter = sol.tree_queries
    max_time = sol.max_time
    max_depth = sol.max_depth

    A = actiontype(pomdp)

    empty!(planner.tree)
    insert_root!(planner.tree, b, sol.n_particles)

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
    if a_idx == 0; a = rand(actions(pomdp)); end

    return a::A, Dict{Symbol, Any}(
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

function Base.empty!(tree::PFTDPWTree)
    empty!(tree.Nh)
    empty!(tree.Nha)
    empty!(tree.Qha)

    empty!(tree.b)
    empty!(tree.b_children)
    empty!(tree.b_rewards)

    empty!(tree.bao_children)
    empty!(tree.ba_children)
    empty!(tree.obs_weights)

    nothing
end

function get_placeholders(pomdp::POMDP{S,A,O}) where {S,A,O}
    a = rand(actions(pomdp))
    s = rand(initialstate(pomdp))
    o = @gen(:o)(pomdp,s,a)
    return a::A, o::O
end
