@info "Loading Dependencies..."
using Pkg
Pkg.activate(@__DIR__)
using Test
using POMDPs, POMDPModelTools, POMDPModels, QuickPOMDPs
using PFTDPW
using SubHunt
using LaserTag
using VDPTag2


@info "Constructing Problems..."
include(join([@__DIR__,"/LightDarkPOMDP.jl"]))

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
