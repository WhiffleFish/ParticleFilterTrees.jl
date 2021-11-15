# PFTDPW
### Particle Filter Trees with Double Progressive Widening

## Parameters
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


## Usage
```julia
using POMDPs, POMDPModels
using PFTDPW

pomdp = LightDark1D()
b0 = initialstate(pomdp)
solver = PFTDPWSolver(tree_queries=10_000, check_repeat_obs=false)
planner = solve(solver, pomdp)
a = action(planner, b0)
```
