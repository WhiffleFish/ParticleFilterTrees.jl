function D3Trees.D3Tree(tree::PFTDPWTree{S,A,O}; show_obs::Bool=true) where {S,A,O}

    if isempty(tree.bao_children)
        show_obs = false
        @warn """show_obs=true, but no observation labels found.
        Make sure check_repeat_obs=true in solver to track observations"""
    end

    n_b = length(tree.b)
    n_ba = length(tree.ba_children)
    children = Vector{Int}[Int[] for _ in 1:(n_b+n_ba)]
    text = Vector{String}(undef, n_b+n_ba)
    tooltip = Vector{String}(undef, n_b+n_ba)
    link_style = Vector{String}(undef, n_b+n_ba)

    a_dict = Dict{Int,A}()
    show_obs && ( o_dict = Dict{Int,O}(bp_idx=>o for ((ba_idx,o),bp_idx) in tree.bao_children) )

    ba_parent = Dict{Int, Int}()
    b_parent = Dict{Int,Int}(())

    for (ba_idx,bp_list) in enumerate(tree.ba_children)
        for bp_idx in bp_list
            b_parent[bp_idx] = ba_idx
        end
    end

    GREY_COLOR = hex(colorant"grey")
    hex_colors = Vector{String}(undef, n_b+n_ba)

    for b_idx in 1:n_b
        hex_colors[b_idx] = GREY_COLOR

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
            ba_parent[ba_idx] = b_idx
        end
        if b_idx == 1
            link_style[b_idx] = ""
            text[b_idx] = """
                <root>
                V=$(round(V;sigdigits=3))"""
        else
            if show_obs
                o = o_dict[b_idx]
                o isa Float64 && (o = round(o;sigdigits=3))
                text[b_idx] = """
                    o = $o
                    N = $N
                    V = $(round(V;sigdigits=3))"""
            else
                text[b_idx] = """
                    N = $N
                    V = $(round(V;sigdigits=3))"""
            end
            Nha = tree.Nha[b_parent[b_idx]]
            stroke_width = link_width(N,Nha)
            link_style[b_idx] = "stroke-width:$(stroke_width)px"
        end
    end

    minQ, maxQ = extrema(tree.Qha)

    for ba_idx in 1:n_ba
        Q = round(tree.Qha[ba_idx], sigdigits=3)
        a = a_dict[ba_idx]
        N = tree.Nha[ba_idx]

        relQ = (tree.Qha[ba_idx]-minQ)/(maxQ-minQ)
        ba_color = weighted_color_mean(relQ, colorant"green", colorant"red");
        hex_colors[n_b+ba_idx] = hex(ba_color);
        text[ba_idx + n_b] = """
            N = $N
            a = $a
            Q = $Q"""

        children[ba_idx + n_b] = tree.ba_children[ba_idx]
        tooltip[ba_idx + n_b] = "ba_idx = $ba_idx"

        stroke_width = link_width(N,tree.Nh[ba_parent[ba_idx]])
        link_style[ba_idx + n_b] = "stroke-width:$(stroke_width)px"
    end

    style = fill("stroke:#",n_b+n_ba) .* hex_colors

    return D3Trees.D3Tree(
        children,
        text = text,
        style = style,
        tooltip = tooltip,
        link_style = link_style,
        title = "PFT-DPW Tree"
    )
end

function link_width(N::Int, maxN::Int; max_width::Int=20)
    return max(round(sqrt(N/maxN), sigdigits=3)*max_width,1)
end
