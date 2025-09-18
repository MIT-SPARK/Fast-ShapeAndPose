## CAST mask video
# Lorenzo Shaikewitz, 9/4/2025

import Images
import GeometryBasics
import FileIO
import JSON
using Serialization
using FastPACE
import Plots
using SimpleRotations
using Printf
using LinearAlgebra

parentimg = "/home/lorenzo/research/tracking/datasets/racecar_offline/slow"


# load data
dets = JSON.parsefile("data/cast/robin_cast.json")
K = convert.(Float64,reduce(hcat,dets[1]["K_mat"])')
data = deserialize("data/cast/SCF.dat")

kpts_test = Dict()
gt_test = Dict()
robin_test = Dict()
for (i,d) in enumerate(dets)
    kpts_test[i] = Dict(1=>convert.(Float64,reduce(hcat,d["est_world_keypoints"])) ./ 1000.)
    T = convert.(Float64,reduce(hcat,d["gt_teaser_pose"]))'
    gt_test[i] = Dict(1=>(T[1:3,1:3], T[1:3,4]/1000.))

    # robin
    robin_test[i] = Dict("time"=>d["robin_time"], "inliers"=>d["robin_inliers"])
end


function plot_frame!(plt, pose, K; gt=false)
    R = pose[1]
    t = pose[2]

    origin = t
    axes   = [t + 0.1*R[:,i] for i in 1:3]  # x, y, z axes in world

    # project to image
    project(p) = (K * (p))[1:2] ./ (K * (p))[3]
    o2d = project(origin)
    a2d = [project(p) for p in axes]

    if gt
        colors = [:darkred, :darkgreen, :darkblue]
    else
        colors = [:red, :green3, :blue]
    end
    for (a, c) in zip(a2d, colors)
        Plots.plot!(plt, [o2d[1], a[1]], [o2d[2], a[2]], color=c, lw=gt ? 3 : 1.5, legend=false)
    end
    return plt
end



anim = Plots.@animate for frame in sort(collect(keys(data["solns"])))
    y = kpts_test[frame][1]
    
    # plot image
    img_name = split(dets[frame]["rgb_image_filename"], "/")[end]
    img = Images.load(parentimg*"/"*img_name)
    plt = Plots.plot(img, axis=false, grid=false, title="$img_name")

    # plot keypoints
    for (idx, kpt) = enumerate(eachcol(y))
        u = Int(round(kpt[2]))
        v = Int(round(kpt[1]))
        Plots.scatter!(plt, [v], [u], ms=1, label=false, c=:gold, msw=0)
    end

    # ground truth
    pose_gt = gt_test[frame][1]
    plot_frame!(plt, pose_gt, K; gt=true)

    # pose estimate
    soln = data["solns"][frame]
    plot_frame!(plt,(soln.R, soln.p), K; gt=false)
end


println("\nSaving...")
Plots.gif(anim, "data/cast.mp4", fps=15)