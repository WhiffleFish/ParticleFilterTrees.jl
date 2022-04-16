@testset "PFTDPW" begin
    ro = RolloutSimulator(max_steps=50)
    qmdp = QMDPSolver()

    pomdp = TigerPOMDP()
    sol = SparsePFTSolver(max_time=0.1, max_depth=20, enable_action_pw=false, check_repeat_obs=true)
    sol_QMDP = SparsePFTSolver(max_time=0.1, max_depth=20, enable_action_pw=false, check_repeat_obs=true, action_selector=qmdp)
    planner = solve(sol, pomdp)
    planner_QMDP = solve(sol_QMDP, pomdp)
    simulate(ro, pomdp, planner, BootstrapFilter(pomdp, 10_000))
    simulate(ro, pomdp, planner_QMDP, BootstrapFilter(pomdp, 10_000))

    pomdp = gen_lasertag()
    sol = SparsePFTSolver(max_time=0.1, max_depth=40, enable_action_pw=false, check_repeat_obs=false)
    sol_QMDP = SparsePFTSolver(max_time=0.1, max_depth=40, enable_action_pw=false, check_repeat_obs=false, action_selector=qmdp)
    planner = solve(sol, pomdp)
    planner_QMDP = solve(sol_QMDP, pomdp)
    simulate(ro, pomdp, planner, BootstrapFilter(pomdp, 10_000))
    simulate(ro, pomdp, planner_QMDP, BootstrapFilter(pomdp, 10_000))


    pomdp = SubHuntPOMDP()
    sol = SparsePFTSolver(max_time=0.1, max_depth=40, enable_action_pw=false, check_repeat_obs=false)
    sol_QMDP = SparsePFTSolver(max_time=0.1, max_depth=40, enable_action_pw=false, check_repeat_obs=false, action_selector=qmdp)
    planner = solve(sol, pomdp)
    planner_QMDP = solve(sol_QMDP, pomdp)
    simulate(ro, pomdp, planner, BootstrapFilter(pomdp, 10_000))
    simulate(ro, pomdp, planner_QMDP, BootstrapFilter(pomdp, 10_000))

    pomdp = VDPTagPOMDP()
    sol = SparsePFTSolver(max_time=0.1, max_depth=40, enable_action_pw=true, check_repeat_obs=false)
    planner = solve(sol, pomdp)
    simulate(ro, pomdp, planner, BootstrapFilter(pomdp, 10_000))
end
