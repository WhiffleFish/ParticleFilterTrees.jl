"""
Insert b node into tree

# Arguments
- `tree:::PFTDPWTree` - Tree into which the belief is inserted
- `b::PFTBelief` - Weighted particles representing belief
- `ba_idx::Int` - index id for belief-action node used to generate new belief
- `obs` - observation received from G(s,a)
- `r::Float64` - reward for going from ba node with obs o to node b
"""
function insert_belief!(tree::PFTDPWTree{S,A,O}, b::PFTBelief{S}, ba_idx::Int, obs::O, r::Float64, planner::PFTDPWPlanner)::Nothing where {S,A,O}
    n_b = length(tree.b)+1
    push!(tree.b, b)
    push!(tree.Nh, 0)
    push!(tree.b_rewards, r)
    push!(tree.ba_children[ba_idx],n_b)
    push!(tree.terminal, isterminalbelief(planner.pomdp,b))

    if planner.sol.enable_action_pw
        push!(tree.b_children, Tuple{A,Int}[])
    else
        push!(tree.b_children, sizehint!(Tuple{A,Int}[],planner._SA))
    end

    if planner.sol.check_repeat_obs
        tree.bao_children[(ba_idx,obs)] = n_b
        push!(tree.obs_weights[ba_idx].values,1)
        tree.obs_weights[ba_idx].sum += 1
    end
    nothing
end

function initial_belief(planner, b, n_p::Int, resample::Bool)
    term_ws = 0.0
    w_i = inv(n_p)

    if resample
        s = get_cached_particles(planner.cache)
        for i in 1:n_p
            s_i = rand(b)
            s[i] = s_i
            !isterminal(planner.pomdp, s_i) && (term_ws += w_i)
        end
        return ResamplingPFTBelief(s, term_ws)

    else
        s,w = get_cached_belief(planner.cache)
        w = fill!(w, w_i)
        for i in 1:n_p
            s_i = rand(b)
            s[i] = s_i
            !isterminal(planner.pomdp, s_i) && (term_ws += w_i)
        end
        return RegPFTBelief(s, w, term_ws)
    end
end

function insert_root!(planner::PFTDPWPlanner, b, n_p::Int)

    particle_b = initial_belief(planner, b, n_p, planner.sol.resample)

    A = actiontype(planner.pomdp)
    push!(planner.tree.b, particle_b)
    push!(planner.tree.b_children, Tuple{A,Int}[])
    push!(planner.tree.Nh, 0)
    push!(planner.tree.b_rewards, 0.0)
    push!(planner.tree.terminal, isterminalbelief(planner.pomdp,particle_b))
    nothing
end

"""
Insert ba node into tree
"""
function insert_action!(planner::PFTDPWPlanner, tree::PFTDPWTree{S,A,O}, b_idx::Int, a::A, check_repeat_obs::Bool)::Nothing where {S,A,O}
    n_ba = length(tree.ba_children)+1
    push!(tree.b_children[b_idx], (a,n_ba))
    if iszero(planner.sol.alpha_o)
        push!(tree.ba_children, sizehint!(Int[],Int(planner.sol.k_o)))
    else
        push!(tree.ba_children, Int[])
    end
    push!(tree.Nha, 0)
    push!(tree.Qha, 0.0)

    if check_repeat_obs
        if iszero(planner.sol.alpha_o)
            tree.obs_weights[n_ba] = StatsBase.weights(
                sizehint!(Int[],Int(planner.sol.k_o))
                )
        else
            tree.obs_weights[n_ba] = StatsBase.weights(Int[])
        end
    end
    nothing
end
