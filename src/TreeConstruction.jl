function insert_belief!(tree::PFTDPWTree{S,A,O}, b::PFTBelief{S}, ba_idx::Int, obs::O, r::Float64, planner::PFTDPWPlanner) where {S,A,O}
    n_b = length(tree.b)+1
    push!(tree.b, b)
    push!(tree.Nh, 0)
    push!(tree.b_rewards, r)
    push!(tree.ba_children[ba_idx],n_b)

    freenext!(tree.b_children)

    if planner.sol.check_repeat_obs
        tree.bao_children[(ba_idx,obs)] = n_b
    end
    return n_b
end

initial_belief(pomdp::POMDP, b, n_p::Int) = initial_belief(Random.GLOBAL_RNG, pomdp, b, n_p)

function initial_belief(rng::AbstractRNG, pomdp::POMDP{S}, b, n_p::Int) where S
    s = Vector{S}(undef, n_p)
    w_i = inv(n_p)
    w = fill(w_i, n_p)
    term_ws = 0.0

    for i in eachindex(s)
        s_i = rand(rng,b)
        s[i] = s_i
        !isterminal(pomdp, s_i) && (term_ws += w_i)
    end

    return PFTBelief(s, w, term_ws)
end

function initialize_belief!(rng::AbstractRNG, s::Vector{S}, w::Vector{Float64}, pomdp::POMDP{S}, b) where {S}
    n_p = length(s)
    w_i = inv(n_p)
    w = fill!(w,w_i)
    term_ws = 0.0

    for i in eachindex(s)
        s_i = rand(rng,b)
        s[i] = s_i
        !isterminal(pomdp, s_i) && (term_ws += w_i)
    end

    return PFTBelief(s, w, term_ws)
end

function insert_root!(planner::PFTDPWPlanner, b)
    s,w = gen_empty_belief(planner.cache, planner.sol.n_particles)
    particle_b = initialize_belief!(planner.sol.rng, s, w, planner.pomdp, b)

    push!(planner.tree.b, particle_b)
    freenext!(planner.tree.b_children)
    push!(planner.tree.Nh, 0)
    push!(planner.tree.b_rewards, 0.0)
    nothing
end

function insert_action!(planner::PFTDPWPlanner, tree::PFTDPWTree{S,A}, b_idx::Int, a::A) where {S,A}
    n_ba = length(tree.ba_children)+1
    push!(tree.b_children[b_idx], (a,n_ba))

    freenext!(tree.ba_children)

    push!(tree.Nha, 0)
    push!(tree.Qha, 0.0)

    return n_ba
end
