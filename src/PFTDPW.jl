module PFTDPW

using ParticleFilters # WeightedParticleBelief
using POMDPSimulators # RolloutSimulator
using POMDPs
using Parameters # @with_kw
using Random # Random.GLOBAL_RNG
using BeliefUpdaters # NothingUpdater
using D3Trees
using POMDPModelTools
import StatsBase
using LinearAlgebra
using RandomNumbers: Xorshifts

export PFTDPWTree, PFTDPWSolver, PFTDPWPlanner, RandomRollout

include("belief.jl")

@with_kw struct PFTDPWTree{S,A,O}
    Nh::Vector{Int} = Int[]
    Nha::Vector{Int} = Int[] # Number of times a history-action node has been visited
    Qha::Vector{Float64} = Float64[] # Map ba node to associated Q value

    b::Vector{PFTBelief{S}} = PFTBelief{S}[]
    b_children::Vector{Vector{Tuple{A,Int}}} = Vector{Tuple{A,Int}}[] # b_idx => [(a,ba_idx), ...]
    b_rewards::Vector{Float64} = Float64[] # Map b' node index to immediate reward associated with trajectory bao where b' = τ(bao)

    bao_children::Dict{Tuple{Int,O},Int} = Dict{Tuple{Int,O},Int}() # (ba_idx,O) => bp_idx
    ba_children::Vector{Vector{Int}} = Vector{Int}[] # ba_idx => [bp_idx, bp_idx, bp_idx, ...]
    obs_weights::Dict{Int,Vector{Int}} = Dict{Int,Vector{Int}}()

    function PFTDPWTree{S,A,O}(sz::Int, check_repeat_obs::Bool) where {S,A,O}
        sz = min(sz, 100_000)
        return new(
            sizehint!(Int[], sz),
            sizehint!(Int[], sz),
            sizehint!(Float64[], sz),

            sizehint!(PFTBelief{S}[], sz),
            sizehint!(Vector{Tuple{A,Int}}[], sz),
            sizehint!(Float64[], sz),

            sizehint!(Dict{Tuple{Int,O},Int}(), check_repeat_obs ? sz : 0),
            sizehint!(Vector{Int}[], sz),
            sizehint!(Dict{Int,Vector{Int}}(), check_repeat_obs ? sz : 0)
            )
    end
end

@with_kw struct PFTDPWSolver{RNG<:AbstractRNG, UPD <:Updater} <: Solver
    max_depth::Int         = 20
    n_particles::Int       = 100
    c::Float64             = 1.0
    k_o::Float64           = 10.0
    alpha_o::Float64       = 0.0 # Observation Progressive widening parameter
    k_a::Float64           = 5.0
    alpha_a::Float64       = 0.0 # Action Progressive widening parameter
    tree_queries::Int      = 1_000
    max_time::Float64      = Inf # (seconds)
    rng::RNG               = Xorshifts.Xoroshiro128Star()
    updater::UPD           = NothingUpdater()
    check_repeat_obs::Bool = true
    enable_action_pw::Bool = false
end

struct RandomRollout{RNG<:AbstractRNG,  A} <: Policy
    rng::RNG
    actions::A
end

RandomRollout(pomdp::POMDP) = RandomRollout(Xorshifts.Xoroshiro128Star(),actions(pomdp))

POMDPs.action(p::RandomRollout,b) = rand(p.rng, p.actions)

struct PFTDPWPlanner{M<:POMDP, SOL<:PFTDPWSolver, TREE<:PFTDPWTree, P<:Policy, A} <: Policy
    pomdp::M
    sol::SOL
    tree::TREE
    rollout_policy::P

    _placeholder_a::A
    _SA::Int # Size of action space (for sizehinting)
end

PFTDPWPlanner(pomdp::POMDP,sol::PFTDPWSolver,tree::PFTDPWTree) = PFTDPWPlanner(pomdp, sol, tree, RandomRollout(pomdp))

include("ProgressiveWidening.jl")
include("Generator.jl")
include("TreeConstruction.jl")
include("search.jl")
include("main.jl")
include("TreeVis.jl")

end # module
