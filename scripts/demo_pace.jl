## Demonstrate SCF on PACE
# Lorenzo Shaikewitz, 6/26/2025

using Printf
using FastPACE

prob, gt, y = genproblem(N = 10, K = 4, σm = 0.05, r = 0.2)


weights = ones(prob.N)
lam = 0.0
out = @timed solvePACE_SCF(prob, y, weights, lam; global_iters=1, local_iters = 250)
soln, obj_val = out.value
time_solve = out.time - out.compile_time

# TODO: can make certifying a lot faster
# (integrate more tightly with solvePACE_SCF)
out = @timed certify_rotmat(prob, soln, y, weights, lam; tol=1e-3)
cert = out.value
time_cert = out.time - out.compile_time
if cert
    @printf "Solution found in %.2f ms, certified in %.2f ms." time_solve*1000 time_cert*1000
end