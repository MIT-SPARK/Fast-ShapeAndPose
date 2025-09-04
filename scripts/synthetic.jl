## Run against other local solvers on synthetic data
# Solvers:
# - SDP
# - Manopt
# - GN
# - LM
# - Power iteration
# - Objective termination
# Measure:
# - estimation error (rotation only)
# - runtime
# - number of iterations (if relevant)
# - optimality certificate (if relevant)
# Lorenzo Shaikewitz, 8/28/2025

# TODO: adapt from old version!

using Serialization
using FastPACE

# parameters
r = 0.2
N = 10
Ks = [4,25]
σms = [0.05, 0.5]
problems_per_config = 1000

# generate all data
# data = gendata(r, N, Ks, σms, problems_per_config)


# SCF + cert
data = gendata(r, N, Ks, σms, problems_per_config)
println("Starting SCF + Certificate...")
errors_scfcert, runtimes_scfcert, iter_scfcert = runlocaliter(data, solvePACE_SCF; certify=true, max_iters = 250)
serialize("data/synthetic/scf_cert.dat", [errors_scfcert, runtimes_scfcert, iter_scfcert])

# SCF local
data = gendata(r, N, Ks, σms, problems_per_config)
println("Starting SCF...")
errors_scf, runtimes_scf, iter_scf = runlocaliter(data, solvePACE_SCF; certify=false, max_iters = 250)
serialize("data/synthetic/scf.dat", [errors_scf, runtimes_scf, iter_scf])

# G-N local
data = gendata(r, N, Ks, σms, problems_per_config)
println("Starting GN...")
errors_gn, runtimes_gn = runlocal(data, solvePACE_GN; max_iters = 250)
serialize("data/synthetic/gn.dat", [errors_gn, runtimes_gn])

# L-M local
data = gendata(r, N, Ks, σms, problems_per_config)
println("Starting LM...")
errors_lm, runtimes_lm = runlocal(data, solvePACE_GN; λ_lm=0.1, max_iters = 250)
serialize("data/synthetic/lm.dat", [errors_lm, runtimes_lm])
# # Manopt local
# println("Starting Manopt...")
# errors_manopt, runtimes_manopt = runlocal(data, solvePACE_Manopt)
# serialize("data/synthetic/manopt.dat", [errors_manopt, runtimes_manopt])
# # SDP global (ish)
# println("Starting SDP...")
# errors_sdp, runtimes_sdp, gaps_sdp = runlocaliter(data, solvePACE_SDP)
# serialize("data/synthetic/sdp.dat", [errors_sdp, runtimes_sdp, gaps_sdp])