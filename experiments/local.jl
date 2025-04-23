## Local solution experiments for masters thesis
# Lorenzo Shaikewitz, 4/16/2025

# utilities
include("local_utils.jl")
using Serialization

# parameters
r = 0.2
N = 10
Ks = [4,25]
σms = LinRange(0, 5*r, 41)
problems_per_config = 1001

# generate all data
data = gendata(r, N, Ks, σms, problems_per_config)


# SCF local
println("Starting SCF...")
errors_scf, runtimes_scf, iter_scf = runlocaliter(data, solvePACE_SCF; global_iters=1, local_iters = 250)
serialize("data/scf.dat", [errors_scf, runtimes_scf, iter_scf])
# SCF local, objective termination
println("Starting SCF obj...")
errors_scfobj, runtimes_scfobj, iter_scfobj = runlocaliter(data, solvePACE_SCF; global_iters=1, obj_thresh=1e-2, local_iters = 250)
serialize("data/scfobj.dat", [errors_scfobj, runtimes_scfobj, iter_scfobj])
# Power local
println("Starting Power...")
errors_power, runtimes_power, iter_power = runlocaliter(data, solvePACE_Power; global_iters=1, local_iters = 500)
serialize("data/power.dat", [errors_power, runtimes_power, iter_power])
# G-N local
println("Starting GN...")
errors_gn, runtimes_gn = runlocal(data, solvePACE_GN)
serialize("data/gn.dat", [errors_gn, runtimes_gn])
# L-M local
println("Starting LM...")
errors_lm, runtimes_lm = runlocal(data, solvePACE_GN; λ=0.1)
serialize("data/lm.dat", [errors_lm, runtimes_lm])
# Manopt local
println("Starting Manopt...")
errors_manopt, runtimes_manopt = runlocal(data, solvePACE_Manopt)
serialize("data/manopt.dat", [errors_manopt, runtimes_manopt])
# SDP global (ish)
println("Starting SDP...")
errors_sdp, runtimes_sdp, gaps_sdp = runlocaliter(data, solvePACE_TSSOS)
serialize("data/sdp.dat", [errors_sdp, runtimes_sdp, gaps_sdp])