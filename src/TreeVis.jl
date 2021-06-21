function D3Tree(tree::PFTDPWTree{S,A,O}) where {S,A,O}

    maxN = max(maximum(tree.Nh[2:end]),maximum(tree.Nha))
    n_b = tree.n_b
    n_ba = tree.n_ba
    children = Vector{Int}[Int[] for _ in 1:(n_b+n_ba)]
    text = Vector{String}(undef, n_b+n_ba)
    tooltip = Vector{String}(undef, n_b+n_ba)
    link_style = Vector{String}(undef, n_b+n_ba)

    a_dict = Dict{Int,A}()
    # o_dict = Dict{Int,Float64}(bp_idx=>o for ((ba_idx,o),bp_idx) in tree.bao_children)

    for b_idx in 1:n_b
        r = round(tree.b_rewards[b_idx], sigdigits=3)
        N = tree.Nh[b_idx]

        ch = tree.b_children[b_idx]
        if !isempty(ch)
            V = maximum(tree.Qha[ba_idx] for (_,ba_idx) in ch)
        else
            V = 0.0
        end


        tooltip[b_idx] = "b_idx = $b_idx\nr=$r"
        for (a, ba_idx) in tree.b_children[b_idx]
            push!(children[b_idx],ba_idx + n_b)
            a_dict[ba_idx] = a
        end
        if b_idx == 1
            link_style[b_idx] = ""
            text[b_idx] = "<root>\nV=$(round(V;sigdigits=3))"
        else
            text[b_idx] = "N = $N\nV = $(round(V;sigdigits=3))"
            stroke_width = link_width(N,maxN)
            link_style[b_idx] = "stroke-width:$(stroke_width)px"
        end
    end

    for ba_idx in 1:n_ba
        Q = round(tree.Qha[ba_idx], sigdigits=3)
        a = a_dict[ba_idx]
        N = tree.Nha[ba_idx]
        text[ba_idx + n_b] = "N = $N \na = $a \nQ = $Q"
        children[ba_idx + n_b] = tree.ba_children[ba_idx]
        tooltip[ba_idx + n_b] = "ba_idx = $ba_idx"

        stroke_width = link_width(N,maxN)
        link_style[ba_idx + n_b] = "stroke-width:$(stroke_width)px"
    end

    return D3Trees.D3Tree(
        children,
        text = text,
        tooltip = tooltip,
        link_style = link_style,
        title = "PFT-DPW Tree"
    )
end

function link_width(N::Int, maxN::Int; max_width::Int=20)
    return max(round(N/maxN, sigdigits=3)*max_width,1)
end
