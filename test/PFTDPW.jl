@testset "PFTDPW" begin
    ro = RolloutSimulator(max_steps=50)
    PO_VE = PFTDPW.PORollout(QMDPSolver(); n_rollouts=1)

    pomdp = TigerPOMDP()
    sol = PFTDPWSolver(max_time=0.1, max_depth=20, enable_action_pw=false, check_repeat_obs=true)
    sol_QMDP = PFTDPWSolver(max_time=0.1, max_depth=20, enable_action_pw=false, check_repeat_obs=true, value_estimator=PO_VE)
    planner = solve(sol, pomdp)
    planner_QMDP = solve(sol_QMDP, pomdp)
    simulate(ro, pomdp, planner, BootstrapFilter(pomdp, 10_000))
    simulate(ro, pomdp, planner_QMDP, BootstrapFilter(pomdp, 10_000))

    pomdp = gen_lasertag()
    sol = PFTDPWSolver(max_time=0.1, max_depth=40, enable_action_pw=false, check_repeat_obs=false)
    sol_QMDP = PFTDPWSolver(max_time=0.1, max_depth=40, enable_action_pw=false, check_repeat_obs=false, value_estimator=PO_VE)
    planner = solve(sol, pomdp)
    planner_QMDP = solve(sol_QMDP, pomdp)
    simulate(ro, pomdp, planner, BootstrapFilter(pomdp, 10_000))
    simulate(ro, pomdp, planner_QMDP, BootstrapFilter(pomdp, 10_000))


    pomdp = SubHuntPOMDP()
    sol = PFTDPWSolver(max_time=0.1, max_depth=40, enable_action_pw=false, check_repeat_obs=false)
    sol_QMDP = PFTDPWSolver(max_time=0.1, max_depth=40, enable_action_pw=false, check_repeat_obs=false, value_estimator=PO_VE)
    planner = solve(sol, pomdp)
    planner_QMDP = solve(sol_QMDP, pomdp)
    simulate(ro, pomdp, planner, BootstrapFilter(pomdp, 10_000))
    simulate(ro, pomdp, planner_QMDP, BootstrapFilter(pomdp, 10_000))

    pomdp = VDPTagPOMDP()
    sol = PFTDPWSolver(max_time=0.1, max_depth=40, enable_action_pw=true, check_repeat_obs=false)
    planner = solve(sol, pomdp)
    simulate(ro, pomdp, planner, BootstrapFilter(pomdp, 10_000))
end
