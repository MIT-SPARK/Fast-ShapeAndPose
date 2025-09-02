## Demonstrate SCF on PACE
# Lorenzo Shaikewitz, 6/26/2025

using Printf
using LinearAlgebra
using SimpleRotations
using FastPACE

# Generate random data
prob, gt, y = genproblem(N = 10, K = 4, σm = 0.01, r = 0.2)
lam = 0.0

# add outlier associations
y = repeat(y, inner=[1,prob.N])
B = repeat(prob.B, outer=[1,prob.N,1])
prob = FastPACE.Problem(size(y,2), prob.K, prob.σm, prob.r, B)

# Run GNC
out = @timed gnc(prob, y, lam, FastPACE.gnc_SCF; cbar2 = 0.05, μUpdate = 1.1)
soln, inliers, success = out.value
time_all = out.time - out.compile_time

@printf "Solution found in %.1f ms with status %s.\n" time_all*1e3 string(success)

# errors
@printf "R error: %.1f°, t error: %.1f mm, c error: %.2f\n" roterror(soln.R, gt.R) norm(soln.p - gt.p)*1000 norm(soln.c - gt.c)

display(inliers)