## Visualize local (and global) convergence of SCF
# Lorenzo Shaikewitz

# Include solvers (TODO: probably should improve this)
include("../src/pace_tools.jl")
import Plots

# projecting quaternions to 3D
function proj_quat(q)
    q_proj = q[2:end]
    if q[1] < 0
        q_proj = -q_proj
    end
    return q_proj
end

# setup
Random.seed!(3) 
# 2x seeds: 7, 8, 14, 20, 24, 27, 31, 32, 41, 46, 54, 56, 60, 67, 78, 84, 85
# 3x seeds: 172, 259
σm = 1.0

# generate problem
prob, gt, y = genproblem(σm=σm)
lam = 0.0
weights = ones(prob.N)

# solve with logs
soln, obj_val, q_logs, ℒ, obj = solvePACE_SCF(prob, y, weights, lam; global_iters=15, logs=true)

function calc_error(q)
    global gt
    

end

# 1) plot eigvals, objective vals
begin
    λs = reduce(hcat,eigvals.(ℒ.(q_logs[1])))[1,:]
    fs = obj.(q_logs[1])

    p1 = Plots.plot(λs .- minimum(λs) .+ 0.01, label=false, ylabel="Min Eigenvalue", yscale=:log10)
    p2 = Plots.plot(fs .+ 0.01, label=false, xlabel="Iteration", ylabel="Objective", yscale=:log10)
    p3 = Plots.plot(p1,p2, layout=(2,1))
end

# 2) plot quaternion trajectory
# begin
#     n = length(q_logs[1])
#     q_projs = reduce(hcat,proj_quat.(q_logs[1]))
#     q_proj_gt = proj_quat(rotm2quat(gt.R))

#     anim = @Plots.animate for cam_i ∈ 1:100
#         i = ceil(Int, cam_i*n / 100)
#         Plots.plot3d(q_projs[1,1:i],q_projs[2,1:i],q_projs[3,1:i], label=false, marker=:circle,
#             seriesalpha=if i > 1 range(0.5, 1, length=i) else 1 end)
#         # ground truth
#         Plots.scatter3d!([q_proj_gt[1]], [q_proj_gt[2]], [q_proj_gt[3]], label="Ground truth", markershape=:cross, ms=10)
        
#         Plots.plot!(aspect_ratio=1,xlim=[-2,2],ylim=[-2,2],zlim=[-2,2], title="Iteration $i", dpi=300,
#             camera = (cam_i/100*90, 30))
#     end
#     Plots.gif(anim, "local_convergence.gif", fps = 15)
# end

# 3) plot global trajectories
# begin
#     q_proj_gt = proj_quat(rotm2quat(gt.R))

#     all_n = []
#     all_q_projs = []
#     for log in q_logs
#         append!(all_n, length(log))
#         push!(all_q_projs, reduce(hcat,proj_quat.(log)))
#     end

#     anim = @Plots.animate for cam_i ∈ 1:100
#         Plots.scatter3d([q_proj_gt[1]], [q_proj_gt[2]], [q_proj_gt[3]], label="Ground truth", markershape=:cross, ms=10)

#         for (idx, q_proj) in enumerate(all_q_projs)
#             i = ceil(Int, cam_i*all_n[idx] / 100)
#             Plots.plot3d!(q_proj[1,1:i],q_proj[2,1:i],q_proj[3,1:i], label=false, marker=:circle,
#                 seriesalpha=if i > 1 range(0.5, 1, length=i) else 1 end)
#         end
        
#         Plots.plot!(aspect_ratio=1,xlim=[-2,2],ylim=[-2,2],zlim=[-2,2], title="Iteration", dpi=300,
#             camera = (cam_i/100*90, 30))
#     end
#     Plots.gif(anim, "global_convergence.gif", fps = 15)
# end

# 4) plot ALL trajectories
# begin
#     soln, obj_val, full_q_logs, ℒ, obj = solvePACE_SCF(prob, y, weights, lam; grid=1000, global_iters=1000, logs=true, all_logs=true)

#     q_proj_gt = proj_quat(rotm2quat(gt.R))

#     all_n = []
#     all_q_projs = []
#     for log in q_logs
#         append!(all_n, length(log))
#         push!(all_q_projs, reduce(hcat,proj_quat.(log)))
#     end

#     full_n = []
#     full_q_projs = []
#     full_coloridx = []
#     for log in full_q_logs
#         append!(full_n, length(log))
#         push!(full_q_projs, reduce(hcat,proj_quat.(log)))

#         coloridx = 1
#         for (idx, q_list) in enumerate(q_logs)
#             q_scf = q_list[end]
#             if abs(abs(log[end]'*q_scf) - 1) < 1e-5
#                 coloridx = idx
#             end
#         end
#         append!(full_coloridx, coloridx)
#     end


#     anim = @Plots.animate for cam_i ∈ 1:100
#         Plots.scatter3d([q_proj_gt[1]], [q_proj_gt[2]], [q_proj_gt[3]], label="Ground truth", markershape=:cross, ms=10, msw=2)

#         for (idx, q_proj) in enumerate(all_q_projs)
#             i = ceil(Int, cam_i*all_n[idx] / 100)
#             Plots.plot3d!(q_proj[1,1:i],q_proj[2,1:i],q_proj[3,1:i], label=false, marker=:circle, ms=2, msw=0.,
#                 seriesalpha=if i > 1 range(0.25, 1, length=i) else 1 end)
#         end

#         for (idx, q_proj) in enumerate(full_q_projs)
#             i = ceil(Int, cam_i*full_n[idx] / 100)
#             Plots.plot3d!(q_proj[1,1:i],q_proj[2,1:i],q_proj[3,1:i], label=false, marker=:circle, ms=2, msw=0.,
#                 seriesalpha=if i > 1 range(0.25, 1, length=i) else 1 end, color=full_coloridx[idx]+1)
#         end
#         Plots.scatter3d!([q_proj_gt[1]], [q_proj_gt[2]], [q_proj_gt[3]], label=false, markershape=:cross, ms=10, msw=2 ,color=1)
        
#         Plots.plot!(aspect_ratio=1,xlim=[-2,2],ylim=[-2,2],zlim=[-2,2], title="Iteration", dpi=300,
#             camera = (cam_i/100*90, 30))
#     end
#     Plots.gif(anim, "all_global_convergence.gif", fps = 15)
# end