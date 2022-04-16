@testset "Non-terminal Sampling" begin
    struct TestPOMDP <: POMDP{Int,Int,Int}
        term_state::Int
    end

    pomdp = TestPOMDP(0)

    POMDPs.isterminal(p::TestPOMDP, s::Int) = s == p.term_state

    b = PFTDPW.PFTBelief([0,2,0,4], Float64[0.20,0.25,0.25,0.25], pomdp)
    rng = Random.GLOBAL_RNG
    N = 50_000
    samples = [PFTDPW.non_terminal_sample(rng, pomdp, b) for _ in 1:N]

    for (s,w) in PFTDPW.weighted_particles(b)
        c = count(==(s), samples)
        ratio = c/N
        expected = isterminal(pomdp, s) ? 0.0 : w/b.non_terminal_ws
        # println("s:$s \t experimental:$(round(ratio,sigdigits=4)) \t expected:$(round(expected,sigdigits=4)) ")
        @test isapprox(ratio, expected, atol=1e-2)
    end
end
