using PFTDPW
using QMDP
using BasicPOMCP
using POMDPs, POMDPSimulators, ParticleFilters, Plots, StatsPlots

hr = HistoryRecorder(max_steps=50)
ro = RolloutSimulator(max_steps=50)
pomdp = LightDark

VE = FOValue(ValueIterationSolver())
VE = PORollout(QMDPSolver(), BootstrapFilter(LightDark, 20))
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
solver = PFTDPWSolver(; PFTDPW_params...)
planner = solve(solver, pomdp)

@profiler a, info = action_info(planner, initialstate(LightDark))
t = D3Tree(info.tree)
inchrome(t)

# r = simulate(ro, pomdp, planner, BootstrapFilter(pomdp, 1_000))
h = simulate(hr, pomdp, planner, BootstrapFilter(pomdp, 1_000))
LDPlot(h)
disc_rew(h, pomdp)

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

##
include(joinpath(@__DIR__,"LightDarkPOMDP.jl"))
pomdp = LightDark

VE = PFTDPW.PORollout(QMDPSolver(), n_rollouts=10)

PFTDPW_params = Dict{Symbol,Any}(
    :c => 100.0,
    :k_o => 4.0,
    :k_a => 4.0,
    :alpha_o => 1/10,
    :alpha_a => 0.0,
    :n_particles => 20,
    :max_depth => 20,
    :tree_queries => 20_000,
    :value_estimator => VE,
    :check_repeat_obs => false,
    :enable_action_pw => false
)
solver = PFTDPWSolver(max_time=1.0; PFTDPW_params...)
planner = solve(solver, pomdp)
a, info = action_info(planner, initialstate(LightDark))

h = simulate(hr, pomdp, planner, BootstrapFilter(pomdp, 10_000))
LDPlot(h)
disc_rew(h, pomdp)


n_sims = 50
sims = @showprogress [simulate(ro, pomdp, planner, BootstrapFilter(pomdp, 1_000)) for _ in 1:n_sims]

mean(sims)
std(sims)/sqrt(50)

@profiler action(planner, initialstate(LightDark)) recur=:flat
