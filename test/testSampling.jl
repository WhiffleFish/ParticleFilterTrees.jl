using POMDPs
using Plots
using PFTDPW
using Random
struct TestPOMDP <: POMDP{Int,Int,Int}
    term_state::Int
end

pomdp = TestPOMDP(2)

POMDPs.isterminal(p::TestPOMDP, s::Int) = s == p.term_state

b = PFTDPW.PFTBelief{Int}([1,2,3,4], fill(1/4,4), 0.75)
rng = Random.GLOBAL_RNG
samples = [PFTDPW.non_terminal_sample(rng, pomdp, b) for _ in 1:10_000]

histogram(samples)
