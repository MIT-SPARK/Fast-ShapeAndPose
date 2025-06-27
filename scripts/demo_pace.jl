## Demonstrate SCF on PACE
# Lorenzo Shaikewitz, 6/26/2025

using FastPACE

prob, gt, y = genproblem(N = 10, K = 4, σm = 0.05, r = 0.2)


weights = ones(prob.N)
lam = 0.0
soln, obj_val = solvePACE_SCF(prob, y, weights, lam; global_iters=1, local_iters = 250)