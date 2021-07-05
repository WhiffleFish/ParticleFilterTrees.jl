using SubHunt
using POMDPSimulators

pomdp = SubHuntPOMDP()

solver = PFTDPWSolver(
    max_time=0.1,
    c = 100,
    k_o = 2.0,
    k_a = 4.0,
    alpha_o = 1/10,
    n_particles = 20,
    max_depth = 50,
    tree_queries = 1_000_000,
    check_repeat_obs = false,
    enable_action_pw=false
)
planner = solve(solver, pomdp)

sim = RolloutSimulator(max_steps = 50)
u = BootstrapFilter(pomdp, 10_000)
simulate(sim, pomdp, planner, u)

@profiler simulate(sim, pomdp, planner, u)
