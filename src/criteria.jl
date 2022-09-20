struct MaxUCB
    c::Float64
    MaxUCB(c=1.0) = new(c)
end

function select_best(criteria::MaxUCB, tree::PFTDPWTree{S,A}, b_idx) where {S,A}
    c = criteria.c
    lnh = log(tree.Nh[b_idx])
    local opt_a::A
    max_ucb = -Inf
    opt_idx = 0

    for (a,ba_idx) in tree.b_children[b_idx]
        Nha = tree.Nha[ba_idx]
        iszero(Nha) && return a::A, ba_idx::Int
        Q̂ = tree.Qha[ba_idx]
        ucb = Q̂ + c*sqrt(lnh / Nha)

        if ucb > max_ucb
            max_ucb = ucb
            opt_a = a
            opt_idx = ba_idx
        end
    end
    return opt_a, opt_idx
end

struct MaxPoly
    c::Float64
    β::Float64
    MaxPoly(c=1.0,β=1/4) = new(c,β)
end

function select_best(criteria::MaxPoly, tree::PFTDPWTree{S,A}, b_idx) where {S,A}
    (;c,β) = criteria
    Nh = tree.Nh[b_idx]
    powNh = Nh^β
    local opt_a::A
    max_ucb = -Inf
    opt_idx = 0

    for (a,ba_idx) in tree.b_children[b_idx]
        Nha = tree.Nha[ba_idx]
        iszero(Nha) && return a::A, ba_idx::Int
        Q̂ = tree.Qha[ba_idx]
        ucb = Q̂ + c*powNh / sqrt(Nha)

        if ucb > max_ucb
            max_ucb = ucb
            opt_a = a
            opt_idx = ba_idx
        end
    end
    return opt_a, opt_idx
end

struct MaxQ end

function select_best(criteria::MaxQ, tree::PFTDPWTree{S,A}, b_idx) where {S,A}
    local opt_a::A
    maxQ = -Inf
    opt_idx = 0

    for (a,ba_idx) in tree.b_children[b_idx]
        Q̂ = tree.Qha[ba_idx]

        if Q̂ > maxQ
            maxQ = Q̂
            opt_a = a
            opt_idx = ba_idx
        end
    end
    return opt_a, opt_idx
end
