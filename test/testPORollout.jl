using POMDPModelTools
using POMDPs
using POMDPSimulators
using PFTDPW
using D3Trees
using JET
using JETTest
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
    violin!((0:(T-1))', [t.b.particles for t in h], labels="", c="red", alpha=0.4)
    plot!(0:(T-1), zeros(T), ls=:dash, lc=:green, label="goal")
    xlabel!("Time Step")
    ylabel!("Position")
end

include(joinpath(@__DIR__,"LightDarkPOMDP.jl"))
pomdp = LightDark

hr = HistoryRecorder(max_steps=50)

VE = PFTDPW.PORollout(QMDPSolver())
PFTDPW_params = Dict{Symbol,Any}(
    :c => 100.0,
    :k_o => 4.0,
    :k_a => 4.0,
    :alpha_o => 1/10,
    :alpha_a => 0.0,
    :n_particles => 20,
    :max_depth => 20,
    :tree_queries => 1_000,
    :value_estimator => VE,
    :check_repeat_obs => false,
    :enable_action_pw => false
)

solver = PFTDPWSolver(max_time=0.1; PFTDPW_params...)
planner = solve(solver, pomdp)

h = simulate(hr, pomdp, planner, BootstrapFilter(pomdp, 1_000))
LDPlot(h)
disc_rew(h, pomdp)
