"""
Return weighted average rollout
"""
function rollout(planner::Policy, solver::Solver, b::WeightedParticleBelief, d::Int)::Float64 # Paralellizable
    r = 0.0
    sim = RolloutSimulator(rng = solver.rng, max_steps = d)
    for (s,w) in weighted_particles(b)
        r += (w/b.weight_sum)*simulate(
                sim,
                planner.pomdp,
                planner.rollout_policy,
                planner.updater,
                b,
                s
            )::Float64
    end
    return r::Float64
end


function search(planner::Policy, sol::PFTDPWSolver, b_idx::Int, d::Int)::Float64
    tree = planner.tree
    pomdp = planner.pomdp

    if d == 0
        return 0.0
    end

    a = act_prog_widen(pomdp, tree, sol, b_idx)
    ba_idx = tree.b_children[b_idx][a]
    if length(tree.ba_children[ba_idx]) <= sol.k_o*tree.Nha[ba_idx]^sol.alpha_o
        bp, o, r = GenBelief(sol.rng, pomdp, tree.b[b_idx], a)

        # MAKE SURE WE'RE NOT OVERWRITING BELIEFS
        # NOTE: This check is not necessary in continuous obs
        if !haskey(tree.ba_children[ba_idx], o)
            insert_belief!(tree, bp, ba_idx, o, r)
        end

        total = r + discount(pomdp)*rollout(planner, sol, bp, d-1)
    else
        o, bp_idx = rand(tree.ba_children[ba_idx])
        r = tree.b_rewards[bp_idx]
        total = r + discount(pomdp)*search(planner, sol, bp_idx, d-1)
    end
    tree.Nh[b_idx] += 1
    tree.Nha[ba_idx] += 1

    tree.Qha[ba_idx] = incremental_avg(tree.Qha[ba_idx], total, tree.Nha[ba_idx])
    # tree.Qha[ba_idx] + (total - tree.Qha[ba_idx])/tree.Nha[ba_idx]

    return total::Float64
end

function POMDPs.solve(pomdp::POMDP{S,A,O}, sol::PFTDPWSolver)::PFTDPWPlanner where {S,A,O}
    return PFTDPWPlanner(pomdp, sol, PFTDPWTree{S,A,O}())
end

function initial_belief(b, n_p::Int)
    if b isa WeightedParticleBelief
        return b
    else
        # rand(b, n_p) doesn't work -> For TigerPOMDP "Sampler not defined for this object"
        s = [rand(b) for _ in 1:n_p]
        w = repeat([1/n_p], n_p)
        return WeightedParticleBelief(s,w)
    end
end

function POMDPModelTools.action_info(planner::PFTDPWPlanner, b)::Dict{Symbol, Any}
    # NOTE: moved actions to beginning of function for more accurate timing
    t0 = time()

    sol = planner.sol
    pomdp = planner.pomdp
    max_iter = sol.tree_queries
    max_time = sol.max_time
    max_depth = sol.max_depth

    S = statetype(pomdp)
    A = actiontype(pomdp)
    O = obstype(pomdp)

    planner.tree = PFTDPWTree{S,A,O}()
    insert_belief!(planner.tree, initial_belief(b, sol.n_particles), 0, first(observations(pomdp)), 0.0)


    iter = 0
    while (time()-t0 < max_time) && (iter < max_iter)
        search(planner, sol, 1, max_depth)
        iter += 1
    end

    a = UCB1action(planner.tree, 1, 0.0)

    return Dict(
        :action => a,
        :n_iter => iter,
        :tree => sol.tree_in_info ? planner.tree : nothing
        )
end

function POMDPs.action(planner::PFTDPWPlanner, b)
    return action_info(planner, b)[:action]
end
