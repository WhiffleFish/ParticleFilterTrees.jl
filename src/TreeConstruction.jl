"""
Insert b node into tree

# Arguments
- `tree:::PFTDPWTree` - Tree into which the belief is inserted
- `b::WeightedParticleBelief` - Weighted particles representing belief
- `ba_idx::Int` - index id for belief-action node used to generate new belief
- `obs` - observation received from G(s,a)
- `r::Float64` - reward for going from ba node with obs o to node b
"""
function insert_belief!(tree::PFTDPWTree{S,A,O}, b::WeightedParticleBelief{S}, ba_idx::Int, obs::O, r::Float64, planner::PFTDPWPlanner)::Nothing where {S,A,O}
    n_b = length(tree.b)+1
    push!(tree.b, b)
    push!(tree.Nh, 0)
    push!(tree.b_rewards, r)
    push!(tree.ba_children[ba_idx],n_b)

    if planner.sol.enable_action_pw
        push!(tree.b_children, Tuple{A,Int}[])
    else
        L = length(actions(planner.pomdp))
        push!(tree.b_children, sizehint!(Tuple{A,Int}[],L))
    end

    if planner.sol.check_repeat_obs
        tree.bao_children[(ba_idx,obs)] = n_b
        push!(tree.obs_weights[ba_idx],1)
    end
    nothing
end

function initial_belief(b, n_p::Int)
    s = [rand(b) for _ in 1:n_p]
    w = fill(inv(n_p), n_p)
    return WeightedParticleBelief(s,w)
end

function insert_root!(tree::PFTDPWTree{S,A,O}, b, n_p::Int)::Nothing where {S,A,O}
    n_b = length(tree.b)+1
    particle_b = initial_belief(b, n_p)

    push!(tree.b, particle_b)
    push!(tree.b_children, Tuple{A,Int}[])
    push!(tree.Nh, 0)
    push!(tree.b_rewards, 0.0)
    nothing
end

"""
Insert ba node into tree
"""
function insert_action!(tree::PFTDPWTree{S,A,O}, b_idx::Int, a::A, check_repeat_obs::Bool)::Nothing where {S,A,O}
    n_ba = length(tree.ba_children)+1
    push!(tree.b_children[b_idx], (a,n_ba))
    push!(tree.ba_children, Int[])
    push!(tree.Nha, 0)
    push!(tree.Qha, 0.0)

    if check_repeat_obs
        tree.obs_weights[n_ba] = Int[]
    end
    nothing
end
