# ParticleFilterTrees.jl
[![CI](https://github.com/WhiffleFish/ParticleFilterTrees.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/WhiffleFish/ParticleFilterTrees.jl/actions/workflows/CI.yml)[![codecov](https://codecov.io/gh/WhiffleFish/ParticleFilterTrees.jl/branch/main/graph/badge.svg?token=9HK0F2AU2R)](https://codecov.io/gh/WhiffleFish/ParticleFilterTrees.jl)


## Particle Filter Trees with Double Progressive Widening

## Parameters

| Parameter         | Type          | Default               | Description |
| ----------------- | ------------- | --------------------- | ------------------------- |
| `max_depth`       | `Int`         | `20`                  | Maximum tree search depth |
| `n_particles`     | `Int`         | `100`                 | Number of particles representing tree node belief |
| `criterion`       | `Any`         | `PolyUCB(1.)`         | Action selection criterion |
| `k_o`             | `Float64`     | `10.0`                | Initial observation widening parameter |
| `alpha_o`         | `Float64`     | `0.0`                 | Observation progressive widening parameter |
| `k_a`             | `Float64`     | `5.0`                 | Initial action widening parameter |
| `alpha_a`         | `Float64`     | `0.0`                 | Action progressive widening parameter |
| `tree_queries`    | `Int`         | `1_000`               | Maximum number of tree search iterations |
| `max_time`        | `Float64`     | `Inf`                 | Maximum tree search time (in seconds) |
| `rng`             | `AbstractRNG` | `Random.default_rng()`| Random number generator |
| `value_estimator` | `Any`         | `FastRandomSolver()`  | Belief node value estimator |
| `check_repeat_obs`| `Bool`        | `true`                | Check that repeat observations do not overwrite beliefs (added dictionary overhead) |
| `enable_action_pw`| `Bool`        | `false`               | Incrementally sample and add actions if true; add all actions at once if false |
| `beliefcache_size`| `Int`         | `1_000`               | Number of particle/weight vectors to cache offline (reduces online array allocations) |
| `treecache_size`  | `Int`         | `1_000`               | Number of belief/action nodes to preallocate in tree (reduces `Base._growend!` calls) |


## Usage

```julia
using POMDPs, POMDPModels
using ParticleFilterTrees

pomdp = LightDark1D()
b0 = initialstate(pomdp)
solver = PFTDPWSolver(tree_queries=10_000, check_repeat_obs=false)
planner = solve(solver, pomdp)
a = action(planner, b0)
```

### SparsePFT

Using sufficiently large `treecache_size` and `beliefcache_size` allows for very few online allocations.

```julia
using BenchmarkTools

pomdp = LightDark1D()
b0 = initialstate(pomdp)
solver = SparsePFTSolver(
    max_time            = 0.1, 
    tree_queries        = 100_000, 
    treecache_size      = 50_000, 
    beliefcache_size    = 50_000, 
    check_repeat_obs    = false
)
planner = solve(solver, pomdp)
action(planner, b0)
@btime action(planner, b0)
```

```
100.006 ms (0 allocations: 0 bytes)
```

