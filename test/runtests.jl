@info "Loading Dependencies..."
using Pkg
Pkg.activate(@__DIR__)
using Test
using POMDPs, POMDPModelTools, POMDPModels, QuickPOMDPs, POMDPSimulators
using StaticArrays
using ParticleFilters
using PFTDPW
using SubHunt
using LaserTag
using VDPTag2


@info "Constructing Problems..."
include(joinpath(@__DIR__,"LightDarkPOMDP.jl"))

tiger = TigerPOMDP()
baby = BabyPOMDP()
lasertag = gen_lasertag()
subhunt = SubHuntPOMDP()
tag = VDPTagPOMDP()


@info "Instantiating Planners..."
tiger_sol = PFTDPWSolver(max_time=0.1, max_depth=20, enable_action_pw=false, check_repeat_obs=true)
tiger_planner = solve(tiger_sol, tiger)

baby_sol = tiger_sol
baby_planner = solve(baby_sol, baby)

lt_sol = PFTDPWSolver(max_time=0.1, max_depth=40, enable_action_pw=false, check_repeat_obs=false)
lt_planner = solve(lt_sol, lasertag)

subhunt_sol = PFTDPWSolver(max_time=0.1, max_depth=40, enable_action_pw=false, check_repeat_obs=false)
subhunt_planner = solve(subhunt_sol, subhunt)

tag_sol = PFTDPWSolver(max_time=0.1, max_depth=40, enable_action_pw=true, check_repeat_obs=false)
tag_planner = solve(tag_sol, tag)


@info "Testing Rollouts..."
ro = RolloutSimulator(max_steps=100)

@show simulate(ro, tiger, tiger_planner, BootstrapFilter(tiger, 1_000))

@show simulate(ro, baby, baby_planner, BootstrapFilter(baby, 1_000))

@show simulate(ro, lasertag, lt_planner, BootstrapFilter(lasertag, 100_000))

@show simulate(ro, subhunt, subhunt_planner, BootstrapFilter(subhunt, 100_000))

@show simulate(ro, tag, tag_planner, BootstrapFilter(tag, 100_000))


## Subhunt terminal state error
@info "SubHunt terminal state error check"

const END_KILL = SubState([-1,-1], [-1,-1], -1, false)
const Pos = SVector{2, Int}

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
push!(subhunt_planner.tree.b_children, Tuple{A,Int}[])
push!(subhunt_planner.tree.Nh, 0)
push!(subhunt_planner.tree.b_rewards, 0.0)

PFTDPW.no_obs_check_search(subhunt_planner, 1, 10)
# ^ would fail to run if isterminalbelief did not take weights into account


## Nonterminal sample test
include("testSampling.jl")


## Performance Testing
include("testPerformance.jl")

@info "Testing Complete"
