@info "Loading Dependencies..."
using Pkg
Pkg.activate(@__DIR__)
using Test
using POMDPs, POMDPModelTools, POMDPModels, QuickPOMDPs
using SubHunt
using LaserTag
using VDPTag2


@info "Constructing Problems..."
const R = 60
const LIGHT_LOC = 10

const LightDark = QuickPOMDP(
    states = -R:R+1, # r+1 is a terminal state
    actions = (-10, -1, 0, 1, 10),
    discount = 0.95,
    isterminal = s::Int -> s==R::Int+1,
    obstype = Float64,

    transition = function (s::Int, a::Int)
        if a == 0
            return Deterministic{Int}(R::Int+1)
        else
            return Deterministic{Int}(clamp(s+a, -R::Int, R::Int))
        end
    end,

    observation = (s, a, sp) -> Normal(sp, abs(sp - LIGHT_LOC::Int) + 1e-3),

    reward = function (s, a, sp, o)
        if a == 0
            return s == 0 ? 100 : -100
        else
            return -1.0
        end
    end,

    initialstate = POMDPModelTools.Uniform(div(-R::Int,2):div(R::Int,2))
)

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


@info "Test action_info"
@inferred action_info(tiger_planner, initialstate(tiger))

@inferred action_info(baby_planner, initialstate(baby))

@inferred action_info(lt_planner, initialstate(lasertag))

@inferred action_info(subhunt_planner, initialstate(subhunt))

@inferred action_info(tag_planner, initialstate(tag))
