## Test region of attraction theory
# Lorenzo Shaikewitz, 4/1/2025

# Include solvers (TODO: probably should improve this)
include("../src/pace_tools.jl")
import Plots

# setup
Random.seed!(14) 
# 2x seeds: 7, 8, 14, 20, 24, 27, 31, 32, 41, 46, 54, 56, 60, 67, 78, 84, 85
# 3x seeds: 172, 259
σm = 1.0

# generate problem
prob, gt, y = genproblem(σm=σm)
lam = 0.0
weights = ones(prob.N)

# solve with logs
soln, obj_val, q_logs, ℒ, obj = solvePACE_SCF(prob, y, weights, lam; global_iters=15, debug=true)

println("Found $(length(q_logs)) solution(s).")

# 1) plot angular distance from minima
angles11 = sin.(acos.([min(1.,q1'*q_logs[1][end]) for q1 in q_logs[1]]))
angles12 = sin.(acos.([q1'*q_logs[2][end] for q1 in q_logs[1]]))
angles21 = sin.(acos.([q1'*q_logs[1][end] for q1 in q_logs[2]]))
angles22 = sin.(acos.([min(1.,q1'*q_logs[2][end]) for q1 in q_logs[2]]))

p1=Plots.plot([angles11 angles12], label=[1 2], ylabel="Angle")
Plots.hline!([min(maximum(angles11), minimum(angles12))], label=false)
p2=Plots.plot([angles21 angles22], label=[1 2], ylabel="Angle", xlabel="Iteration")
Plots.hline!([min(maximum(angles22), minimum(angles21))], label=false)
plot_one =Plots.plot(p1,p2, layout=(2,1))

# 2) find region of attraction by looking at all runs
soln, obj_val, full_q_logs, ℒ, obj = solvePACE_SCF(prob, y, weights, lam; global_iters=1000, debug=true, all_logs=true)

angles(log, qstar) = sin.(acos.([min(1.,q'*qstar) for q in log]))

all_sines = []
Δs = zeros(length(full_q_logs), length(full_q_logs))
for (i1, log1) in enumerate(full_q_logs)
    for (i2, log2) in enumerate(full_q_logs)
        sines = angles(log1, log2[end])
        if i1 == i2
            push!(all_sines, sines)
        end
        # max if local mins are close
        if abs(abs(log1[end]'log2[end]) - 1) < 1e-5
            Δ = maximum(sines)
        else
            Δ = minimum(sines)
        end
        Δs[i1, i2] = Δ
    end
end
Δs_prelim = minimum(eachrow(Δs))

ids = zeros(Int, length(full_q_logs))
Δs_collected = ones(length(q_logs))
for (i1, log1) in enumerate(full_q_logs)
    for (i2, log2) in enumerate(q_logs)
        if abs(abs(log1[end]'log2[end]) - 1) < 1e-5
            Δs_collected[i2] = min(Δs_collected[i2], Δs_prelim[i1])
            ids[i1] = i2
            break
        end
    end
end
objs = [obj(log[end]) for log in q_logs]
display([Δs_collected objs])

# 3) Plot quat
function proj_quat(q)
    q_proj = q[2:end]
    if q[1] < 0
        q_proj = -q_proj
    end
    return q_proj
end

p2 = Plots.plot()
for (idx, q_log) in enumerate(full_q_logs)
    local qs = reduce(hcat, proj_quat.(q_log))
    Plots.scatter3d!(p2, qs[1,:], qs[2,:], qs[3,:], label=false, aspect_ratio=:equal, color=ids[idx])
end
Plots.scatter3d!(p2,aspect_ratio=1,xlim=[-2,2],ylim=[-2,2],zlim=[-2,2])
solns = []
for log in q_logs
    push!(solns, proj_quat(log[end]))
end
solns = reduce(hcat, solns)
Plots.plot!(p2, solns[1,:], solns[2,:], solns[3,:], seriestype=:scatter, color=:white, markershape=:cross, ms=10, label="local min")


# 4) color the sphere by distance to some point
# projecting quaternions to 3D
function view_sphere()
    # sample 1000 points
    samples = normalize.(eachcol(randn(4,1000)))
    samples_projected = reduce(hcat, proj_quat.(samples))

    # compute angular distance
    sample_angles = angles(samples, samples[1])

    # plot
    c = Plots.cgrad(:inferno)
    Plots.plot(eachrow(samples_projected)..., seriestype=:scatter, color=c[max.(1,ceil.(Int,256*sample_angles))])
end

function view_sphere_obj()
    # sample 1000 points
    samples = normalize.(eachcol(randn(4,1000)))
    samples_projected = reduce(hcat, proj_quat.(samples))

    # compute obj
    sample_objs = obj.(samples)

    # plot
    Plots.plot(eachrow(samples_projected)..., seriestype=:scatter, marker_z = sample_objs, label=false, title="Distance to objective", color=:bluesreds)
    Plots.plot!(solns[1,:], solns[2,:], solns[3,:], seriestype=:scatter, color=:white, markershape=:cross, ms=10, label="local min")
end