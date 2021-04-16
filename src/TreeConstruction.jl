"""
Insert b node into tree

# Arguments
- `tree:::PFTDPWTree` - Tree into which the belief is inserted
- `b::WeightedParticleBelief` - Weighted particles representing belief
- `ba_idx::Int` - index id for belief-action node used to generate new belief
- `obs` - observation received from G(s,a)
- `r::Float64` - reward for going from ba node with obs o to node b
"""
function insert_belief!(tree::PFTDPWTree{S,A,O}, b::WeightedParticleBelief{S}, ba_idx::Int, obs::O, r::Float64)::Nothing where {S,A,O}
    # NOTE: Parent ba_idx of root node is 0
    tree.n_b += 1
    push!(tree.b, b)
    push!(tree.b_children, Dict{A, Int}())
    push!(tree.Nh, 0)
    push!(tree.b_parent, ba_idx) # root node has parent ba_idx 0
    push!(tree.b_rewards, r)

    # root node doesn't have associated reaching action/observation
    if !(ba_idx == 0)
        tree.ba_children[ba_idx][obs] = tree.n_b
    end
    nothing
end


"""
Insert ba node into tree
"""
function insert_action!(tree::PFTDPWTree{S,A,O}, b_idx::Int, a::A)::Nothing where {S,A,O}
    tree.n_ba += 1
    tree.b_children[b_idx][a] = tree.n_ba
    push!(tree.ba_children, Dict{O,Int}())
    push!(tree.ba_parent, b_idx)
    push!(tree.Nha, 0)
    push!(tree.Qha, 0.0)
    nothing
end
