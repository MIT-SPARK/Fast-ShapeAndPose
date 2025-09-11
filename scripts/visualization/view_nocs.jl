## View NOCS keypoints for debugging
# Lorenzo Shaikewitz, 9/4/2025

import Images
import JSON
using Glob

parentimg = "/home/lorenzo/research/playground/catkeypoints"
imgframe = 13+3

# load data
det_files = glob("data/nocs/mug/*.json")
det_file = det_files[1]
dets = JSON.parsefile(det_file)
y = convert.(Float64,reduce(hcat, dets[imgframe]["est_pixel_keypoints"]))
img_name = dets[imgframe]["rgb_image_filename"]

inliers = [3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

# plot image
img = Images.load(parentimg*"/"*img_name)
plt = Plots.plot(img, axis=false, grid=false, title="$img_name")

# plot keypoints
for (idx, kpt) = enumerate(eachcol(y))
    u = Int(round(kpt[2]))
    v = Int(round(kpt[1]))
    Plots.scatter!(plt, [v], [u], ms=1, label=false, c=:gold, msw=0)
    if idx in inliers
        Plots.scatter!(plt, [v], [u], ms=1, label=false, c=:blue, msw=0)
    end
end

K = [591.0125 0. 322.525; 0 590.165 244.11084; 0 0 1]

function plot_frame!(plt, pose, K; gt=false)
    R = pose[1]
    t = pose[2]

    origin = t
    axes   = [t + 0.1*R[:,i] for i in 1:3]  # x, y, z axes in world

    # project to image
    project(p) = (K * (p))[1:2] ./ (K * (p))[3]
    o2d = project(origin)
    a2d = [project(p) for p in axes]

    colors = [:red, :green, :blue]
    for (a, c) in zip(a2d, colors)
        Plots.plot!(plt, [o2d[1], a[1]], [o2d[2], a[2]], color=c, lw=gt ? 2 : 1, legend=false)
    end
    return plt
end
# ground truth
# T = convert.(Float64, reduce(hcat, dets[imgframe]["gt_pose"]))'
# pose = (T[1:3,1:3], T[1:3,4])
pose = gt_all["scene_1-mug_daniel_norm.json"][imgframe][1]
plot_frame!(plt, pose, K; gt=true)

# pose estimate
data = deserialize("data/nocs/mug/SCF.dat")
soln = data["solns"][split(det_file,"/")[end]][imgframe]
plot_frame!(plt,(soln.R, soln.p), K; gt=false)

# 3D plot of keypoints?
# y2 = convert.(Float64,reduce(hcat, dets[imgframe]["est_world_keypoints"]))
# y2 = y2[:,y2[3,:] .!= 0]
# Plots.scatter(eachrow(y2)...)