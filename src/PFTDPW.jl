module PFTDPW

using POMDPs
using Random # Random.GLOBAL_RNG
using D3Trees # TreeVis
using POMDPModelTools # action_info
using LinearAlgebra # normalize!
using RandomNumbers: Xorshifts # Fast RNG
using Colors # TreeVis
using BasicPOMCP
import MCTS: convert_estimator, estimate_value, convert_to_policy
using PushVectors
using ParticleFilters

export PFTDPWTree, PFTDPWSolver, PFTPlanner

export PFTBelief

include(joinpath("util","belief.jl"))
include(joinpath("util","NestedPushVectors.jl"))

export FastRandomSolver, FastRandomRolloutEstimator

include(joinpath("util","FastBootstrapFilter.jl"))
include(joinpath("util","ValueEstimation.jl"))


abstract type AbstractPFTSolver <: Solver end
abstract type AbstractPFTPlanner <: Policy end

"""
...
- `Nh` - Number of times history node has been visited
- `Nha` - Number of times action node has been visited
- `Qha` - Q value associated with some action node
- `b` - Vector of beliefs (`PFTBelief`)
- `b_children` - Mapping belief ID to (action, action ID) tuple
- `b_rewards` - R(b,a) where index is ID of b' where b' = τ(b,a,o)
- `bao_children` - `(ba_idx,O) => bp_idx`
- `ba_children` - `ba_idx => [bp_idx, bp_idx, bp_idx, ...]`
...
"""
struct PFTDPWTree{S,A,O}
    Nh::PV{Int}
    Nha::PV{Int}# Number of times a history-action node has been visited
    Qha::PV{Float64} # Map ba node to associated Q value

    b::PV{PFTBelief{S}}
    b_children::NPV{Tuple{A,Int}}# b_idx => [(a,ba_idx), ...]
    b_rewards::PV{Float64}# Map b' node index to immediate reward associated with trajectory bao where b' = τ(bao)

    bao_children::Dict{Tuple{Int,O},Int} # (ba_idx,O) => bp_idx
    ba_children::NPV{Int} # ba_idx => [bp_idx, bp_idx, bp_idx, ...]

    function PFTDPWTree{S,A,O}(sz::Int, check_repeat_obs::Bool=true, k_o=10, k_a=10) where {S,A,O}
        return new(
            PushVector{Int}(sz),
            PushVector{Int}(sz),
            PushVector{Float64}(sz),

            PushVector{PFTBelief{S}}(sz),
            NestedPushVector{Tuple{A,Int}}(ceil(Int,k_o), sz),
            PushVector{Float64}(sz),

            sizehint!(Dict{Tuple{Int,O},Int}(), check_repeat_obs ? sz : 0),
            NestedPushVector{Int}(ceil(Int,k_a), sz)
            )
    end
end

"""
...
- `max_depth::Int = 20` - Maximum tree search depth
- `n_particles::Int = 100` - Number of particles representing belief
- `c::Float64 = 1.0` - UCB exploration parameter
- `k_o::Float64 = 10.0` - Initial observation widening parameter
- `k_a::Float64 = 5.0` - Initial action widening parameter
- `tree_queries::Int = 1_000` - Maximum number of tree search iterations
- `max_time::Float64 = Inf` - Maximum tree search time (in seconds)
- `rng::RNG = Xorshifts.Xoroshiro128Star()` - Random number generator
- `action_selector::AS = FastRandomSolver()` - Belief node first action selector
- `enable_action_pw::Bool = false` - Alias for `alpha_a = 0.0`
- `check_repeat_obs::Bool = true` - Check that repeat observations do not overwrite beliefs (added dictionary overhead)
- `beliefcache_size::Int = 100_000` - Number of particle/weight vectors to cache offline
- `treecache_size::Int = 100_000` - Number of belief/action nodes to preallocate in tree (reduces `Base._growend!` calls)
...
"""
Base.@kwdef struct SparsePFTSolver{RNG<:AbstractRNG, AS} <: AbstractPFTSolver
    tree_queries::Int      = 1_000
    max_time::Float64      = Inf # (seconds)
    max_depth::Int         = 20
    n_particles::Int       = 100
    c::Float64             = 1.0
    k_o::Float64           = 10.0
    k_a::Float64           = 5.0
    rng::RNG               = Xorshifts.Xoroshiro128Star()
    action_selector::AS    = FastRandomSolver()
    enable_action_pw::Bool = false
    check_repeat_obs::Bool = true
    beliefcache_size::Int  = 100_000
    treecache_size::Int    = 100_000
end


"""
...
- `max_depth::Int = 20` - Maximum tree search depth
- `n_particles::Int = 100` - Number of particles representing belief
- `c::Float64 = 1.0` - UCB exploration parameter
- `k_o::Float64 = 10.0` - Initial observation widening parameter
- `alpha_o::Float64 = 0.0` - Observation progressive widening parameter
- `k_a::Float64 = 5.0` - Initial action widening parameter
- `alpha_a::Float64 = 0.0` - Action progressive widening parameter
- `tree_queries::Int = 1_000` - Maximum number of tree search iterations
- `max_time::Float64 = Inf` - Maximum tree search time (in seconds)
- `rng::RNG = Xorshifts.Xoroshiro128Star()` - Random number generator
- `value_estimator::VE = FastRandomSolver()` - Belief node value estimator
- `check_repeat_obs::Bool = true` - Check that repeat observations do not overwrite beliefs (added dictionary overhead)
- `enable_action_pw::Bool = false` - Alias for `alpha_a = 0.0`
- `beliefcache_size::Int = 100_000` - Number of particle/weight vectors to cache offline
- `treecache_size::Int = 100_000` - Number of belief/action nodes to preallocate in tree (reduces `Base._growend!` calls)
...
"""
Base.@kwdef struct PFTDPWSolver{RNG<:AbstractRNG, VE} <: AbstractPFTSolver
    tree_queries::Int      = 1_000
    max_time::Float64      = Inf # (seconds)
    max_depth::Int         = 20
    n_particles::Int       = 100
    c::Float64             = 1.0
    k_o::Float64           = 10.0
    alpha_o::Float64       = 0.0 # Observation Progressive widening parameter
    k_a::Float64           = 5.0
    alpha_a::Float64       = 0.0 # Action Progressive widening parameter
    rng::RNG               = Xorshifts.Xoroshiro128Star()
    value_estimator::VE    = FastRandomSolver()
    check_repeat_obs::Bool = true
    enable_action_pw::Bool = false
    beliefcache_size::Int  = 100_000
    treecache_size::Int    = 100_000
end

include(joinpath("util","cache.jl"))

struct PFTDPWPlanner{SOL<:AbstractPFTSolver, M<:POMDP, TREE<:PFTDPWTree, VE, A, S, T} <: AbstractPFTPlanner
    pomdp::M
    sol::SOL
    tree::TREE
    solved_VE::VE

    _placeholder_a::A
    obs_req::T
    cache::BeliefCache{S}
end

struct SparsePFTPlanner{SOL<:AbstractPFTSolver, M<:POMDP, TREE<:PFTDPWTree, AS, A, S, T} <: AbstractPFTPlanner
    pomdp::M
    sol::SOL
    tree::TREE
    solved_action_selector::AS

    _placeholder_a::A
    obs_req::T
    cache::BeliefCache{S}
end

include("ProgressiveWidening.jl")
include("Generator.jl")
include("TreeConstruction.jl")
include("search.jl")
include("main.jl")
include("TreeVis.jl")

end # module
