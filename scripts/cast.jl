## Run on CAST dataset with GNC.
# Measure:
# - pose estimate
# - total runtime
# - GNC iterations
# - optimality certificate
# Lorenzo Shaikewitz, 8/29/2025

# TODO: NEED A GNC IMPLEMENTATION OF SCF

import JSON
using MAT

using FastPACE

# load keypoint data
dets = JSON.parsefile("data/cast.json")
camK = convert.(Float64,reduce(hcat,dets[1]["K_mat"])')

kpts_test = Dict()
gt_test = Dict()
for (i,d) in enumerate(dets)
    kpts_test[i] = Dict(1=>convert.(Float64,reduce(hcat,d["est_world_keypoints"])) ./ 1000.)
    T = convert.(Float64,reduce(hcat,d["gt_teaser_pose"]))'
    gt_test[i] = Dict(1=>(T[1:3,1:3], T[1:3,4]/1000.))
end

# load CAD frame
file = matopen("data/racecar_lib.mat")
shapes = read(file, "shapes") # 3 x N x K
close(file)

# create problem
prob = FastPACE.Problem(size(shapes,2), size(shapes,3), 0.05, 0.2, shapes)

## solve
λ = 0.

times = -ones(length(keys(kpts_test)),2)
errR = -ones(length(keys(kpts_test)),2)
errp = -ones(length(keys(kpts_test)),2)
for (i,frames) in enumerate(sort(keys(kpts_test)))
    y = kpts_test[1]

    # solve with SCF
    out = @timed solvePACE_SCF(prob, y, weights, lam; certify=false, max_iters = 250)
    soln, opt, status, scf_iters = out.value
    time_scf = out.time - out.compile_time
    times[i,1] = time_scf


    # solve with GN
    out = @timed solvePACE_GN(prob, y, weights, lam; max_iters = 250)
    soln, opt = out.value
    time_gn = out.time - out.compile_time
    times[i,2] = time_gn

    if mod(frame, 10) == 0
        print("$frame ")
    end
end