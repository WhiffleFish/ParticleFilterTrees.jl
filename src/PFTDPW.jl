using ParticleFilters # WeightedParticleBelief
using POMDPSimulators # RolloutSimulator
using POMDPs
using Parameters # @with_kw
using Random # Random.GLOBAL_RNG

@with_kw mutable struct PFTDPWTree{A,O}
    Nh::Vector{Int} = Int[]
    Nha::Vector{Int} = Int[] # Number of times a history-action node has been visited
    Qha::Vector{Float64} = Float64[] # Map ba node to associated Q value

    b::Vector{WeightedParticleBelief} = WeightedParticleBelief[]
    b_children::Vector{Dict{A,Int}} = Dict{A, Int}[] # Map belief node index to dict mapping action to belief-action node index
    b_parent::Vector{Int} = Int[] # Map b node index to parent ba node index
    b_rewards::Vector{Float64} = Float64[] # Map b' node index to immediate reward associated with trajectory bao where b' = τ(bao)

    ba_children::Vector{Dict{O,Int}} = Dict{O,Int}[] # Map belief-action node index to dict mapping observation to belief node index
    ba_parent::Vector{Int} = Int[] # Map ba node index to parent b node index

    n_b::Int = 0
    n_ba::Int = 0
end

@with_kw struct PFTDPWSolver <: Solver
    max_depth::Int = 20
    c::Float64 = 1.0
    k_o::Float64 = 10.0
    alpha_o::Float64 = 0.0 # Observation Progressive widening parameter
    k_a::Float64 = 5.0
    alpha_a::Float64 = 0.0 # Action Progressive widening parameter
    tree_queries::Int = 1_000
    max_time::Float64 = Inf
    rng::AbstractRNG = Random.GLOBAL_RNG
end

struct PFTDPWPlanner <: Policy
    pomdp::POMDP
    sol::PFTDPWSolver
    tree::PFTDPWTree
end
