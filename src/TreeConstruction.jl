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

initial_belief(pomdp::POMDP, b, n_p::Int) = initial_belief(Random.GLOBAL_RNG, pomdp, b, n_p)

function initial_belief(rng::AbstractRNG, pomdp::POMDP, b, n_p::Int)
    s = Vector{statetype(pomdp)}(undef, n_p)
    w_i = inv(n_p)
    w = fill(w_i, n_p)
    term_ws = 0.0

    for i in 1:n_p
        s_i = rand(rng,b)
        s[i] = s_i
        !isterminal(pomdp, s_i) && (term_ws += w_i)
    end

    return PFTBelief(s, w, term_ws)
end

function insert_root!(rng::AbstractRNG, tree::PFTDPWTree{S,A,O}, pomdp::POMDP, b, n_p::Int) where {S,A,O}
    particle_b = initial_belief(rng, pomdp, b, n_p)

    push!(tree.b, particle_b)
    push!(tree.b_children, Tuple{A,Int}[])
    push!(tree.Nh, 0)
    push!(tree.b_rewards, 0.0)
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
