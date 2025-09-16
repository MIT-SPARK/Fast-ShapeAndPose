## Plot synthetic keypoints and estimate

using Plots
using Printf
using FastPACE

# Generate random data
prob, gt, y = genproblem(N = 10, K = 4, σm = 0.05, r = 0.2)

weights = ones(prob.N)
lam = 0.0

# You can solve and certify in one call
out = @timed solvePACE_SCF(prob, y, weights, lam; certify=false, max_iters=10)
soln, opt, status, scf_iters = out.value
time_all = out.time - out.compile_time

@printf "Solution found in %.1f μs with status %s.\n" time_all*1e6 string(status)

# plot keypoints
Plots.scatter(eachrow(y)..., label="Measurements", ms=8)
y_est = soln.R*reshape(reshape(prob.B, 3*prob.N, prob.K)*soln.c, (3,prob.N)) .+ soln.p

Plots.scatter!(eachrow(y_est)..., label="Pose est.", ms=8)