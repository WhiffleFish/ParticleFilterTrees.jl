using Pkg
Pkg.activate(@__DIR__)
try Pkg.rm("LaserTag") catch end
try Pkg.rm("SubHunt") catch end
try Pkg.rm("VDPTag2") catch end

Pkg.add(url = "https://github.com/JuliaPOMDP/LaserTag.jl#master")
Pkg.add(url = "https://github.com/WhiffleFish/SubHunt.jl#master")
Pkg.add(url = "https://github.com/zsunberg/VDPTag2.jl#master")
