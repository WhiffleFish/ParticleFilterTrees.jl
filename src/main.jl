"""
Return weighted average rollout
"""
function rollout(planner::PFTDPWPlanner, b::PFTBelief, d::Int)::Float64
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

function POMDPs.solve(sol::PFTDPWSolver, pomdp::POMDP{S,A,O})::PFTDPWPlanner where {S,A,O}
    act = actions(pomdp)
    a = rand(act)

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

    sz = min(100_000, sol.tree_queries)

    if sol.resample
        cache = Cache{S}(
            Vector{S}(undef, sol.n_particles),
            Vector{Float64}(undef, sol.n_particles),
            [Vector{S}(undef, sol.n_particles) for _ in 1:sz],
            Vector{Float64}[], # resampled particles all have equal weight
            sz,
            sol.n_particles,
            0
        )
        tree = PFTDPWTree{S,A,O,ResamplingPFTBelief{S}}(sol.tree_queries, sol.check_repeat_obs)
    else
        cache = Cache{S}(
            S[],
            Float64[],
            [Vector{S}(undef, sol.n_particles) for _ in 1:sz],
            [Vector{Float64}(undef, sol.n_particles) for _ in 1:sz],
            sz,
            sol.n_particles,
            0
        )
        tree = PFTDPWTree{S,A,O,RegPFTBelief{S}}(sol.tree_queries, sol.check_repeat_obs)
    end

    return PFTDPWPlanner(
        pomdp,
        sol,
        tree,
        RandomRollout(pomdp),
        a,
        SA,
        cache
    )
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
    free!(planner.cache)
    insert_root!(planner, b, sol.n_particles)

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
    empty!(tree.obs_weights)

    nothing
end
