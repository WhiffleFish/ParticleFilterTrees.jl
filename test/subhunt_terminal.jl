const END_KILL = SubState([-1,-1], [-1,-1], -1, false)
const Pos = SVector{2, Int}

@testset "SubHunt Terminal" begin
    subhunt = SubHuntPOMDP()
    S = statetype(subhunt)
    A = actiontype(subhunt)
    O = obstype(subhunt)
    subhunt_sol = PFTDPWSolver(
        max_time=0.1,
        max_depth=10,
        enable_action_pw=false,
        check_repeat_obs=false
    )
    subhunt_planner = solve(subhunt_sol, subhunt)
    empty!(subhunt_planner.tree)

    p = fill(END_KILL, 10)
    p[2] = rand(initialstate(subhunt))
    w = fill(1/10,10)
    w[2] = 0.0
    test_b = PFTDPW.PFTBelief{SubState}(p, w, 0.0)

    push!(subhunt_planner.tree.b, test_b)
    PFTDPW.freenext!(subhunt_planner.tree.b_children)
    push!(subhunt_planner.tree.Nh, 0)
    push!(subhunt_planner.tree.b_rewards, 0.0)

    @test try
        PFTDPW.no_obs_check_search(subhunt_planner, 1, 10)
        true
    catch e
        println(e)
        false
    end
    # ^ would fail to run if isterminalbelief did not take weights into account
end
