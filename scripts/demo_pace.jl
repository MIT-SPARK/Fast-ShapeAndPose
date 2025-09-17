## Demonstrate SCF on PACE
# Lorenzo Shaikewitz, 6/26/2025

using Printf
using FastPACE
using SimpleRotations

# Generate random data
prob, gt, y = genproblem(N = 10, K = 4, σm = 0.01, r = 0.2)

weights = ones(prob.N)
λ = 0.0

# Example: Gauss-Newton
out = @timed FastPACE.solvePACE_GN(prob, y, weights, λ)
time_gn = out.time - out.compile_time

@printf "[Gauss-Newton] solution found in %.1f μs.\n" time_gn*1e6

# Example: SCF
# You can solve and certify in one call
out = @timed solvePACE_SCF(prob, y, weights, λ; certify=true, max_iters=10)
soln, opt, status, scf_iters = out.value
time_all = out.time - out.compile_time

@printf "[SCF] Solution found in %.1f μs with status %s.\n" time_all*1e6 string(status)
# this time comparison is not entirely fair
# Julia precompiles the problem data, so the second call has a runtime boost.

# You can also certify separately:
# begin
#     out = @timed solvePACE_SCF(prob, y, weights, λ; certify=false)
#     soln2, opt2, status2, scf_iters2 = out.value
#     time_solve = out.time - out.compile_time

#     # now certify
#     out = @timed certify_rotmat(prob, soln, y, weights, λ; tol=1e-3)
#     cert = out.value
#     time_cert = out.time - out.compile_time
# end

# Example: plot iterates
using Plots
# Plots.plotlyjs() # uncomment for interactive plots
soln, obj_val, q_logs, l, obj = FastPACE.solvePACE_SCF2(prob, y, weights, λ; global_iters=1, debug=true)

n = length(q_logs[1])
q_projs = reduce(hcat,proj_quat.(q_logs[1]))
q_proj_gt = proj_quat(rotm2quat(gt.R))

# ground truth
Plots.scatter3d([q_proj_gt[1]], [q_proj_gt[2]], [q_proj_gt[3]], label="Ground truth", markershape=:diamond, color=:black, ms=6, msw=2)
# trajectory
Plots.plot3d!(q_projs[1,:],q_projs[2,:],q_projs[3,:], marker_z=1:size(q_projs,2), label=false, marker=:circle, msw=0.5, seriescolor=cgrad(:redblue, size(q_projs,2), categorical=true))
Plots.plot!(aspect_ratio=1,xlim=[-2,2],ylim=[-2,2],zlim=[-2,2], title="SCF Iterates")