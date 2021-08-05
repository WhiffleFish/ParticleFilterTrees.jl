#=
Testing:
    Tiger
    Baby
    LightDark
    LaserTag
    SubHunt
    VDPTag2
=#

using Distributed
using Statistics
using DataFrames
using CSV
using Dates

# RESULTS
SIMS = String[]
MEAN_REWARDS = Float64[]
STD_ERR = Float64[]

if PROCS > 1
    worker_ids = Distributed.addprocs(PROCS; exeflags="--project=$(@__DIR__)")
end

@everywhere begin
    using POMDPs
    using ParticleFilters
    using POMDPSimulators
    using POMDPModels

    using QuickPOMDPs
    using Distributions
    using POMDPModelTools
    # include(joinpath(@__DIR__,"LightDarkPOMDP.jl")) # LightDark defined in runtests.jl
    using LaserTag
    using SubHunt
    using VDPTag2
    using PFTDPW
end

include("benchmark.jl")

tiger = TigerPOMDP()
baby = BabyPOMDP()
lasertag = gen_lasertag()
subhunt = SubHuntPOMDP()
tag = VDPTagPOMDP()

@info "Tiger Test"
tiger_solver = PFTDPWSolver(
    max_time = MAX_TIME,
    tree_queries = 1_000_000,
    k_o = 10,
    max_depth = 50,
    c = 100.0,
    n_particles = 100,
    enable_action_pw = false,
    check_repeat_obs = true
)
tiger_planner = solve(tiger_solver, tiger)
tiger_updater = BootstrapFilter(tiger, 1_000)
tiger_r = benchmark(tiger, tiger_updater, tiger_planner, N_SIMS, 100)
push!(SIMS, "Tiger")
tiger_m = mean(tiger_r)
tiger_stderr = std(tiger_r)/sqrt(N_SIMS)
push!(MEAN_REWARDS, tiger_m)
push!(STD_ERR, tiger_stderr)
println("Mean Reward: $(round(tiger_m, sigdigits=4))")
println("StdErr: $(round(tiger_stderr, sigdigits=4))")

@info "Baby Test"
baby_solver = PFTDPWSolver(
    max_time = MAX_TIME,
    tree_queries = 1_000_000,
    k_o = 10,
    max_depth = 50,
    c = 100.0,
    n_particles = 100,
    enable_action_pw = false,
    check_repeat_obs = true
)
baby_planner = solve(baby_solver, baby)
baby_updater = BootstrapFilter(baby, 1_000)
baby_r = benchmark(baby, baby_updater, baby_planner, N_SIMS, 100)
push!(SIMS, "Baby")
baby_m = mean(baby_r)
baby_stderr = std(baby_r)/sqrt(N_SIMS)
push!(MEAN_REWARDS, baby_m)
push!(STD_ERR, baby_stderr)
println("Mean Reward: $(round(baby_m, sigdigits=4))")
println("StdErr: $(round(baby_stderr, sigdigits=4))")

@info "LightDark Test"
lightdark_solver = PFTDPWSolver(
    max_time = MAX_TIME,
    tree_queries = 1_000_000,
    k_o = 10.0,
    alpha_o = 1/20,
    k_a = 4,
    max_depth = 50,
    c = 100.0,
    n_particles = 100,
    enable_action_pw = false,
    check_repeat_obs = false
)
lightdark_planner = solve(lightdark_solver, LightDark)
lightdark_updater = BootstrapFilter(LightDark, 10_000)
lightdark_r = benchmark(LightDark, lightdark_updater, lightdark_planner, N_SIMS, 100)
push!(SIMS, "LightDark")
lightdark_m = mean(lightdark_r)
lightdark_stderr = std(lightdark_r)/sqrt(N_SIMS)
push!(MEAN_REWARDS, lightdark_m)
push!(STD_ERR, lightdark_stderr)
println("Mean Reward: $(round(lightdark_m, sigdigits=4))")
println("StdErr: $(round(lightdark_stderr, sigdigits=4))")

@info "LaserTag Test"
lasertag_solver = PFTDPWSolver(
    max_time = MAX_TIME,
    tree_queries = 1_000_000,
    c = 26.0,
    k_o = 4.0,
    alpha_o = 1/35,
    n_particles = 20,
    max_depth = 69,
    check_repeat_obs = false,
    enable_action_pw = false
)
lasertag_planner = solve(lasertag_solver, lasertag)
lasertag_updater = BootstrapFilter(lasertag, 500_000)
lasertag_r = benchmark(lasertag, lasertag_updater, lasertag_planner, N_SIMS, 100)
push!(SIMS, "LaserTag")
lasertag_m = mean(lasertag_r)
lasertag_stderr = std(lasertag_r)/sqrt(N_SIMS)
push!(MEAN_REWARDS, lasertag_m)
push!(STD_ERR, lasertag_stderr)
println("Mean Reward: $(round(lasertag_m, sigdigits=4))")
println("StdErr: $(round(lasertag_stderr, sigdigits=4))")

@info "SubHunt Test"
subhunt_solver = PFTDPWSolver(
    max_time = MAX_TIME,
    tree_queries=1_000_000,
    c = 100.0,
    k_o = 2.0,
    alpha_o = 1/10,
    n_particles = 20,
    max_depth = 50,
    check_repeat_obs = false,
    enable_action_pw = false
)
subhunt_planner = solve(subhunt_solver, subhunt)
subhunt_updater = BootstrapFilter(subhunt, 100_000)
subhunt_r = benchmark(subhunt, subhunt_updater, subhunt_planner, N_SIMS, 100)
push!(SIMS, "SubHunt")
subhunt_m = mean(subhunt_r)
subhunt_stderr = std(subhunt_r)/sqrt(N_SIMS)
push!(MEAN_REWARDS, subhunt_m)
push!(STD_ERR, subhunt_stderr)
println("Mean Reward: $(round(subhunt_m, sigdigits=4))")
println("StdErr: $(round(subhunt_stderr, sigdigits=4))")


@info "VDPTag Test"
vdptag_solver = PFTDPWSolver(
    max_time = MAX_TIME,
    tree_queries=1_000_000,
    c = 100.0,
    k_o = 8.0,
    k_a = 20.0,
    alpha_o = 1/85,
    alpha_a = 1/25,
    n_particles = 20,
    max_depth = 50,
    check_repeat_obs = false,
    enable_action_pw = true
)
vdptag_planner = solve(vdptag_solver, tag)
vdptag_updater = BootstrapFilter(tag, 100_000)
vdptag_r = benchmark(tag, vdptag_updater, vdptag_planner, N_SIMS, 100)
push!(SIMS, "VDPTag")
vdptag_m = mean(vdptag_r)
vdptag_stderr = std(vdptag_r)/sqrt(N_SIMS)
push!(MEAN_REWARDS, vdptag_m)
push!(STD_ERR, vdptag_stderr)
println("Mean Reward: $(round(vdptag_m, sigdigits=4))")
println("StdErr: $(round(vdptag_stderr, sigdigits=4))")

PROCS > 1 && Distributed.rmprocs(worker_ids)

df = DataFrame(prob=SIMS, mean=MEAN_REWARDS, std_err=STD_ERR)
date_str = Dates.format(now(), "_yyyy-mm-dd")
filename = "results"*date_str*".csv"
filepath = joinpath(@__DIR__, "perf", filename)
CSV.write(filepath,df)
