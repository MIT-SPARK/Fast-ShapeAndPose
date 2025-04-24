## Solve simulated PACE problems with quaternion and SDP solvers.
# Lorenzo Shaikewitz

# Include solvers (TODO: probably should improve this)
include("../src/pace_tools.jl")


# For benchmarking
using BenchmarkTools

function printresults(errors)
    @printf " %d\n" length(errors)
    @printf " (%.3f, %.3f)\n" minimum(errors) maximum(errors)
    @printf " Mean: %.3f\n" mean(errors)
    @printf " Med.: %.3f\n" median(errors)
end

"""
    setup(σm)

Setup a basic problem given noise standard deviation `σm`.
Mostly for benchmarking.
"""
function setup(σm, K=4)
    prob, gt, y = genproblem(σm=σm, K=K)
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

printstyled("SCF Local", underline=true)
errors_local = []
bm_local = @benchmark wrapper!($errors_local, $solvePACE_SCF, data...; global_iters=1) setup=(data=setup(σm))
printresults(errors_local)

printstyled("Gauss-Newton Local", underline=true)
errors_gn = []
bm_gn = @benchmark wrapper!($errors_gn, $solvePACE_GN, data...) setup=(data=setup(σm))
printresults(errors_gn)

printstyled("L-M Local", underline=true)
errors_lm = []
bm_lm = @benchmark wrapper!($errors_lm, $solvePACE_GN, data...; λ=0.1) setup=(data=setup(σm))
printresults(errors_gn)

# printstyled("SCF Global", underline=true)
# errors_global = []
# bm_global = @benchmark wrapper!($errors_global, $solvePACE_SCF, data...; global_iters=15) setup=(data=setup(σm))
# printresults(errors_global)

# printstyled("SCF Obj Termination", underline=true)
# errors_obj = []
# bm_obj = @benchmark wrapper!($errors_obj, $solvePACE_SCF, data...; global_iters=1, obj_thresh=1e-2) setup=(data=setup(σm))
# printresults(errors_obj)
# # computing the objective is computationally expensive, making this slower

# printstyled("Power Method Local", underline=true)
# errors_power = []
# bm_power = @benchmark wrapper!($errors_power, $solvePACE_Power, data...; global_iters=1) setup=(data=setup(σm))
# printresults(errors_power)

printstyled("TSSOS", underline=true)
errors_tssos = []
bm_tssos = @benchmark wrapper!($errors_tssos, $solvePACE_TSSOS, data...) setup=(data=setup(σm))
printresults(errors_tssos)

printstyled("Manopt", underline=true)
errors_manopt = []
bm_manopt = @benchmark wrapper!($errors_manopt, $solvePACE_Manopt, data...) setup=(data=setup(σm))
printresults(errors_manopt)