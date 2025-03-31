## Solve simulated PACE problems with quaternion and SDP solvers.
# Lorenzo Shaikewitz

# Include solvers (TODO: probably should improve this)
include("../src/pace_tools.jl")


# For benchmarking
using BenchmarkTools

function printresults(errors)
    @printf "⋅ (%.3f, %.3f)\n" minimum(errors) maximum(errors)
    @printf "⋅ Mean: %.3f\n" mean(errors)
    @printf "⋅ Med.: %.3f\n" median(errors)
end

"""
    setup(σm)

Setup a basic problem given noise standard deviation `σm`.
Mostly for benchmarking.
"""
function setup(σm)
    prob, gt, y = genproblem(σm=σm)
    lam = 0.0
    weights = ones(prob.N)
    return prob, gt, y, weights, lam
end

"""
    wrapper!(results, solver, prob, gt, y, weights, lam)
    
Solve and save rotation error for generic shape & pose solver.
Saving results adds ~5 μs.
"""
function wrapper!(results, solver, prob, gt, y, weights, lam; kwargs...)
    out = solver(prob, y, weights, lam; kwargs...)
    soln = out[1]
    _, R_err = rotm2axang(gt.R'*soln.R); R_err *= 180/π
    append!(results, R_err)
end

## Run benchmarks!
σm = 0.1

printstyled("SCF Local\n", underline=true)
errors_local = []
bm_local = @benchmark wrapper!($errors_local, $solvePACE_SCF, data...; global_iters=1) setup=(data=setup(σm))
printresults(errors_local)

printstyled("SCF Global\n", underline=true)
errors_global = []
bm_global = @benchmark wrapper!($errors_global, $solvePACE_SCF, data...; global_iters=15) setup=(data=setup(σm))
printresults(errors_global)

printstyled("SCF Obj Termination\n", underline=true)
errors_obj = []
bm_obj = @benchmark wrapper!($errors_obj, $solvePACE_SCF, data...; global_iters=1, objthresh=1e-5) setup=(data=setup(σm))
printresults(errors_obj)

printstyled("Power Method Local\n", underline=true)
errors_power = []
bm_power = @benchmark wrapper!($errors_power, $solvePACE_Power, data...; global_iters=1) setup=(data=setup(σm))
printresults(errors_power)

printstyled("TSSOS\n", underline=true)
errors_tssos = []
bm_tssos = @benchmark wrapper!($errors_tssos, $solvePACE_TSSOS, data...) setup=(data=setup(σm))
printresults(errors_tssos)

printstyled("Manopt\n", underline=true)
errors_manopt = []
bm_manopt = @benchmark wrapper!($errors_manopt, $solvePACE_Manopt, data...) setup=(data=setup(σm))
printresults(errors_manopt)