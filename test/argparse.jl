using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--sims", "-s"
            help = "Number of rollouts to run for each pomdp"
            arg_type = Int
            default = 100
        "--n_procs"
            help = "Number of processes to use for parallel performance sims"
            arg_type = Int
            default = 1
        "--time"
            help = "Time alotted for each planning step (in seconds)"
            arg_type = Float64
            default = 0.1
        "--perf"
            help = "Run performance tests"
            action = :store_true
    end

    return parse_args(s)
end
