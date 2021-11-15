using POMDPModelTools
using POMDPs
using POMDPSimulators
using PFTDPW
using QMDP
using ParticleFilters
using Plots, StatsPlots
function disc_rew(h::SimHistory, pomdp::POMDP)
    r = 0.0
    disc = 1.0
    for t in h
        r += disc*t.r
        disc *= discount(pomdp)
    end
    r
end

function LDPlot(h::SimHistory)
    T = length(h)
    plot(0:(T-1),[t.s for t in h], label="", lw=5, xticks = 0:(T-1))
    scatter!(0:(T-1),[t.s for t in h], label="True State", ms=5)
    violin!(transpose((0:(T-1))), [t.b.particles for t in h], labels="", c="red", alpha=0.4)
    plot!(0:(T-1), zeros(T), ls=:dash, lc=:green, label="goal")
    xlabel!("Time Step")
    ylabel!("Position")
end

include(joinpath(@__DIR__,"LightDarkPOMDP.jl"))
pomdp = LightDark

hr = HistoryRecorder(max_steps=50)
ro = RolloutSimulator(max_steps=50)

VE = PFTDPW.PORollout(QMDPSolver(), n_rollouts=1)
PFTDPW_params = Dict{Symbol,Any}(
    :c => 100.0,
    :k_o => 4.0,
    :k_a => 4.0,
    :alpha_o => 1/10,
    :alpha_a => 0.0,
    :n_particles => 20,
    :max_depth => 20,
    :tree_queries => 50_000,
    :value_estimator => VE,
    :check_repeat_obs => false,
    :enable_action_pw => false
)

solver = PFTDPWSolver(max_time=0.1; PFTDPW_params...)
planner = solve(solver, pomdp)

@profiler a, info = action_info(planner, initialstate(pomdp)) recur=:flat

h = simulate(hr, pomdp, planner, BootstrapFilter(pomdp, 10_000))
LDPlot(h)
disc_rew(h, pomdp)

@progress rew1 = [simulate(ro, pomdp, planner, BootstrapFilter(pomdp, 1000)) for _ in 1:100]

@progress rew2 = [simulate(ro, pomdp, planner, BootstrapFilter(pomdp, 1000)) for _ in 1:100]

violin([rew1, rew2], label="")
mean(rew1)
mean(rew2)


##
using VDPTag2
pomdp = VDPTagPOMDP()
solver = PFTDPWSolver(
    max_time=1.0,
    tree_queries=100_000,
    enable_action_pw=true,
    check_repeat_obs=false,
    treecache_size=0)

planner = solve(solver, pomdp)
b0 = initialstate(pomdp)
@profiler action(planner, b0)

using POMCPOW
p_sol = POMCPOWSolver(max_time=1.0, tree_queries=100_000)
p_plan = solve(p_sol, pomdp)
@profiler action(p_plan, b0)
