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

    if sol.resample
        cache = ReweightCache{S}(
            particle_cache = Vector{S}(undef, sol.n_particles),
            weight_cache = Vector{Float64}(undef, sol.n_particles),
            ap = Vector{Float64}(undef, sol.n_particles),
            alias = Vector{Int}(undef, sol.n_particles),
            larges = Vector{Int}(undef, sol.n_particles),
            smalls = Vector{Int}(undef, sol.n_particles)
        )
        tree = PFTDPWTree{S,A,O,ResamplingPFTBelief{S}}(sol.tree_queries, sol.check_repeat_obs)
    else
        cache = ReweightCache{S}()
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
    insert_root!(planner.tree, pomdp, b, sol.n_particles)

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
