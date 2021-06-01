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
    # NOTE: Parent ba_idx of root node is 0
    tree.n_b += 1
    push!(tree.b, b)
    push!(tree.b_children, Tuple{A,Int}[])
    push!(tree.Nh, 0)
    push!(tree.b_rewards, r)
    push!(tree.ba_children[ba_idx],tree.n_b)

    if planner.sol.check_repeat_obs
        tree.bao_children[(ba_idx,obs)] = tree.n_b
    end
    nothing
end

function initial_belief(b, n_p::Int)
    if b isa WeightedParticleBelief
        return b
    else
        # rand(b, n_p) doesn't work -> For TigerPOMDP "Sampler not defined for this object"
        s = [rand(b) for _ in 1:n_p]
        w = fill(1/n_p, n_p)
        return WeightedParticleBelief(s,w)
    end
end

function insert_root!(tree::PFTDPWTree{S,A,O}, b, n_p::Int)::Nothing where {S,A,O}
    tree.n_b += 1
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
function insert_action!(tree::PFTDPWTree{S,A,O}, b_idx::Int, a::A)::Nothing where {S,A,O}
    tree.n_ba += 1
    push!(tree.b_children[b_idx], (a,tree.n_ba))
    push!(tree.ba_children, Int[])
    push!(tree.Nha, 0)
    push!(tree.Qha, 0.0)
    nothing
end
