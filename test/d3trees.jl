@testset "d3trees" begin
    pomdp = LightDarkPOMDP()
    sol = PFTDPWSolver(
        tree_queries=100_000,
        max_time=10.,
        max_depth=20,
        enable_action_pw=false,
        check_repeat_obs=false,
        criterion = ParticleFilterTrees.MaxUCB(10.),
        resample = true
    )

    planner = solve(sol, pomdp)
    a = action(planner, initialstate(pomdp))

    @test rand(planner.tree.b[1]) isa Int
    @test D3Tree(planner.tree; show_obs=false) isa D3Tree
end
