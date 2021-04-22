function D3Tree(tree::PFTDPWTree; kwargs...)::D3Tree
    action_offset = tree.n_b
    children = Vector{Int}[]
    text = String[]
    tooltips = String[]

    # NOTE: Maybe set up dictionaries this way from the start?
    action_dict = Dict{Int, Any}()
    obs_dict = Dict{Int, Any}()

    for b_idx in 1:tree.n_b
        b_children = Int[]
        for (a,ba_idx) in tree.b_children[b_idx]
            push!(b_children, ba_idx+action_offset)
            action_dict[ba_idx] = a
        end
        push!(children, b_children)
        Nh = tree.Nh[b_idx]
        r = round(tree.b_rewards[b_idx], sigdigits = 3)
        push!(tooltips, "node: $b_idx\nr = $(r)")
    end

    for ba_idx in 1:tree.n_ba
        ba_children = Int[]
        for (o,b_idx) in tree.ba_children[ba_idx]
            push!(ba_children, b_idx)
            obs_dict[b_idx] = o
        end
        push!(children, ba_children)
        push!(text, "a = $(action_dict[ba_idx])\nQ = $(round(tree.Qha[ba_idx],sigdigits=3))")
        Nha = tree.Nha[ba_idx]
        push!(tooltips, "Nha = $Nha")
    end

    belief_labels = ["1"]
    for b_idx in 2:tree.n_b
        Nh = tree.Nh[b_idx]
        push!(belief_labels, "N: $Nh\nobs: $(obs_dict[b_idx])")
    end

    text = vcat(belief_labels, text)

    return D3Tree(children, text=text, tooltip=tooltips, kwargs...)
end
