## Demonstrate SCF on PACE
# Lorenzo Shaikewitz, 6/26/2025

using Printf
using FastPACE

# Generate random data
prob, gt, y = genproblem(N = 10, K = 4, σm = 0.05, r = 0.2)

weights = ones(prob.N)
lam = 0.0

# You can solve and certify in one call
out = @timed solvePACE_SCF(prob, y, weights, lam; certify=true)
soln, opt, status, scf_iters = out.value
time_all = out.time - out.compile_time

@printf "Solution found in %.1f μs with status %s." time_all*1e6 string(status)

# You can also certify separately:
# begin
#     out = @timed solvePACE_SCF(prob, y, weights, lam; certify=false)
#     soln2, opt2, status2, scf_iters2 = out.value
#     time_solve = out.time - out.compile_time

#     # now certify
#     out = @timed certify_rotmat(prob, soln, y, weights, lam; tol=1e-3)
#     cert = out.value
#     time_cert = out.time - out.compile_time
# end