## Solve simulated PACE problems with quaternion and SDP solvers.
# Lorenzo Shaikewitz

# Include solvers (TODO: probably should improve this)
include("../src/pace_tools.jl")


## For benchmarking
using BenchmarkTools

"""
    setup(σm)

Setup a basic problem given noise standard deviation `σm`.
Mostly for benchmarking.
"""
function setup(σm)
    prob, gt, y = genproblem(σm=σm)
    lam = 0.0
    weights = ones(prob.N)
    return prob, y, weights, lam
end

σm = 1.0

@printf "TSSOS:\n"
bm_tssos = @benchmark solvePACE_TSSOS(data...) setup=(data = setup(σm))

@printf "Manopt:\n"
bm_manopt = @benchmark solvePACE_Manopt(data...) setup=(data = setup(σm))

@printf "SCF (local):\n"
bm_local = @benchmark solvePACE_SCF(data...; global_iters=1) setup=(data = setup(σm))

@printf "SCF (global):\n"
bm_global = @benchmark solvePACE_SCF(data...; global_iters=15) setup=(data = setup(σm))