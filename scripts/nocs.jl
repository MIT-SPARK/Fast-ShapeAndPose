## Run on NOCS-REAL275 dataset with GNC.
# Measure:
# - pose estimate
# - shape estimate!
# - total runtime
# - GNC iterations
# - optimality certificate
# Lorenzo Shaikewitz, 8/28/2025

using ArgParse
using LinearAlgebra
using Statistics
using Serialization
using Glob
import JSON
using MAT
using Printf
import Plots
using StatsPlots
using SimpleRotations

using FastPACE

## Command-line arguments
s = ArgParseSettings()
@add_arg_table s begin
    "--force"
        help = "rerun even if data already saved"
        action = :store_true
    "method"
        help = "method to run: SCF, GN, or all. Can call multiple like [SCF,GN] (no spaces)"
        default = "all"
    "object"
        help = "object to test on: mug, laptop, etc."
        default = "mug"
end
parsed_args = parse_args(ARGS, s)
# parsed_args["force"] = true

# don't re-run methods unless force has been called
object = parsed_args["object"]
methods = parsed_args["method"]
methods = isa(methods, Vector) ? methods : [methods]
if methods[1] == "all"
    methods = ["SCF", "GN", "LM", "Manopt", "SCFopt", "SDP"]
end
methods_to_run = []
if parsed_args["force"]
    methods_to_run = methods
else
    for m in methods
        if isfile("data/nocs/$object/$m.dat")
            println("Using data in `data/nocs/$object` for $m.")
        else
            push!(methods_to_run, m)
        end
    end
end

## load data
# load keypoint data
# det_files = glob("data/nocs/$object/scene_*.json")
det_files = glob("data/nocs/$object/robin_scene_*.json")
kpts_all = Dict()
gt_all = Dict()
robin_all = Dict()
for file in det_files
    dets = JSON.parsefile(file)

    kpts_test = Dict()
    gt_test = Dict()
    robin_test = Dict()
    for (i,d) in enumerate(dets)
        kpts_test[i] = Dict(1=>convert.(Float64,reduce(hcat, d["est_world_keypoints"]))) # [m]
        T = convert.(Float64, reduce(hcat, d["gt_pose"]))'
        # gt_test[i] = Dict(1=>(T[1:3,1:3], T[1:3,4]))

        # fix the ground truth (to align models)
        # rotate R by 90 deg about x axis
        Roffset = axang2rotm([1.;0;0],-π/2)
        # Roffset = diagm(ones(3))

        gt_test[i] = Dict(1=>(project2SO3(T[1:3,1:3])*Roffset, T[1:3,4]))

        # robin
        robin_test[i] = Dict("time"=>d["robin_time"], "inliers"=>d["robin_inliers"])
    end
    # push!(kpts_all, kpts_test)
    # push!(gt_all, gt_test)
    json = split(file,"/")[end]
    kpts_all[json] = kpts_test
    gt_all[json] = gt_test
    robin_all[json] = robin_test
end

# CAD frame
file = matopen("data/nocs/$object/shapes.mat")
shapes = read(file, "shapes") # 3 x N x K
close(file)

# solve!
if !isempty(methods_to_run)
    λ = 0.

    times       = Dict(Pair.(methods_to_run, [Dict([Pair(split(file,"/")[end], Dict()) for file in det_files]) for i in methods_to_run]))
    gnc_success = Dict(Pair.(methods_to_run, [Dict([Pair(split(file,"/")[end], Dict()) for file in det_files]) for i in methods_to_run]))
    gnc_iters = Dict(Pair.(methods_to_run, [Dict([Pair(split(file,"/")[end], Dict()) for file in det_files]) for i in methods_to_run]))
    solns       = Dict(Pair.(methods_to_run, [Dict([Pair(split(file,"/")[end], Dict()) for file in det_files]) for i in methods_to_run]))
    for (id, file) in enumerate(det_files)
        json = split(file,"/")[end]
        println("\n-----$json ($(length(keys(kpts_all[json]))) frames)-----")
        kpts_test = kpts_all[json]
        gt_test = gt_all[json]
        robin_test = robin_all[json]

        for frame in sort(collect(keys(kpts_test)))
            y = kpts_test[frame][1]
            
            # remove robin points
            points_filter = (y[3,:] .!= 0)
            if length(robin_test[frame]["inliers"]) > 0
                robin_filter = zeros(Bool, size(y,2))
                robin_filter[robin_test[frame]["inliers"] .+ 1] .= true
                points_filter = points_filter .&& robin_filter
            end

            # remove occluded / outlier points
            shapes_adj = shapes[:, points_filter,:]
            y = y[:,points_filter]
            if size(y,2) < 3
                # skip frame
                continue
            end
            # create problem
            prob = FastPACE.Problem(size(shapes_adj,2), size(shapes_adj,3), 0.05, 0.2, shapes_adj)

            for method in methods_to_run
                if method == "SDP"
                    out = @timed gnc(prob, y, λ, solvePACE_SDP; cbar2 = 0.005, μUpdate = 1.4)
                    soln, inliers, success, iters = out.value
                elseif method == "Manopt"
                    out = @timed gnc(prob, y, λ, solvePACE_Manopt; cbar2 = 0.005, μUpdate = 1.4)
                    soln, inliers, success, iters = out.value
                elseif method == "GN"
                    out = @timed gnc(prob, y, λ, solvePACE_GN; cbar2 = 0.005, μUpdate = 1.4)
                    soln, inliers, success, iters = out.value
                elseif method == "LM"
                    out = @timed gnc(prob, y, λ, solvePACE_GN; λ_lm=0.1, cbar2 = 0.005, μUpdate = 1.4)
                    soln, inliers, success, iters = out.value
                elseif method == "SCF"
                    out = @timed gnc(prob, y, λ, solvePACE_SCF; certify=false, cbar2 = 0.005, μUpdate = 1.4)
                    soln, inliers, success, iters = out.value
                elseif method == "SCFopt"
                    out = @timed gnc(prob, y, λ, solvePACE_SCF; certify=true, cbar2 = 0.005, μUpdate = 1.4)
                    soln, inliers, success, iters = out.value
                else
                    error("Method $method not implemented.")
                end

                gnc_success[method][json][frame] = success
                gnc_iters[method][json][frame] = iters
                time_dif = out.time - out.compile_time
                # add ROBIN time
                time_dif += robin_test[frame]["time"]
                times[method][json][frame] = time_dif
                solns[method][json][frame] = soln
            end
        

            if mod(frame, 10) == 0
                print("$frame ")
            end
        end
    end

    # save
    for method in methods_to_run
        save_dict = Dict("times"=>times[method], "gnc_success"=>gnc_success[method], 
                "solns"=>solns[method], "gnc_iters"=>gnc_iters[method])
        serialize("data/nocs/$(parsed_args["object"])/$method.dat", save_dict)

        # save just solns in JSON format
        solns_json = Dict()
        for key in keys(solns[method])
            key2 = split(key,".")[1]
            solns_json[key2] = Dict{Int,Any}()
            for frame in keys(solns[method][key])
                solns_json[key2][frame] = Dict("p"=>vec(solns[method][key][frame].p), 
                    "R"=>vec(solns[method][key][frame].R), "c"=>vec(solns[method][key][frame].c))
            end
        end
        open("data/nocs/$(parsed_args["object"])/$method.json","w") do f
            JSON.print(f, solns_json)
        end
    end
end

normalize_frames = [-1,-1,-1,-1,-1,-1]  # pick a frame to normalize by

R_perturb = diagm(ones(3))
# if object == "camera"
#     println("Using perturbation!")
#     R_perturb = project2SO3([0.9999563966163845 -0.0009024000466379273 -0.009294651157040201; 0.0007795913020592626 0.9999124666630083 -0.013207999443500179; 0.009305756464522641 0.01320017750083563 0.9998695705993703])
# end

# visualize!
time_plot = Plots.plot()
jsons = sort(collect(keys(kpts_all)))
for (i,method) in enumerate(methods)
    data = deserialize("data/nocs/$(parsed_args["object"])/$method.dat")
    solns = data["solns"]

    errR = [[roterror(R_perturb'*solns[key1][key2].R, gt_all[key1][key2][1][1]) for key2 in keys(solns[key1])] for key1 in keys(solns)]
    errR = reduce(vcat, errR)
    errp = [[norm(solns[key1][key2].p - gt_all[key1][key2][1][2]) for key2 in keys(solns[key1])] for key1 in keys(solns)]
    errp = reduce(vcat, errp)

    times_dict = data["times"]
    times = reduce(vcat,collect.(values.(values(times_dict))))

    gnc_dict = data["gnc_iters"]
    gnc_iters = reduce(vcat,collect.(values.(values(gnc_dict))))

    println("-----$method-----")
    @printf "5deg5cm: %.1f%%\n" (sum((errR .<= 5) .&& (errp .< 0.05))/length(errR)*100)
    @printf "Rerr: %.1f deg (<0.1m)\n" (mean(errR[errp .< 0.1]))
    @printf "perr: %.1f cm  (<0.1m)\n" (mean(errp[errp .< 0.1])*100)
    @printf "GNCi: %.1f\n" (mean(gnc_iters))
    @printf "Time: %.2f ms\n" (mean(times)*1000)

end

## VERSION WITH APPROX TRANSFORM
# this controls for shape alignment issues
# we select one frame and normalize our poses based on that frame.
println("Transformed results:")
solns_scf = deserialize("data/nocs/$(parsed_args["object"])/SCF.dat")["solns"]

for (i,method) in enumerate(methods)
    data = deserialize("data/nocs/$(parsed_args["object"])/$method.dat")
    solns = data["solns"]

    errR = []
    errp = []
    for video in keys(solns)
        if occursin("canon_len", video)
            video_ref = "robin_scene_4-camera_canon_len_norm.json"
            frame = 123
        elseif occursin("canon_wo", video)
            video_ref = "robin_scene_5-camera_canon_wo_len_norm.json"
            frame = 242
        elseif occursin("shengjun", video)
            video_ref = "robin_scene_2-camera_shengjun_norm.json"
            frame = 448
        elseif occursin("mug_brown", video)
            video_ref = "robin_scene_6-mug_brown_starbucks_norm.json"
            frame = 103
        elseif occursin("anastasia", video)
            video_ref = "robin_scene_6-mug_anastasia_norm.json"
            frame = 166
        elseif occursin("mug_daniel", video)
            video_ref = "robin_scene_2-mug_daniel_norm.json"
            frame = 365
        end
        T1 = [solns_scf[video_ref][frame].R solns_scf[video_ref][frame].p; 0 0 0 1.]
        T2 = [gt_all[video_ref][frame][1][1] gt_all[video_ref][frame][1][2]; 0 0 0 1.]

        T_dif = T2*inv(T1)
        R_dif = T_dif[1:3,1:3]
        p_dif = T_dif[1:3,4]

        errR_video = [roterror(R_dif*solns[video][key2].R, gt_all[video][key2][1][1]) for key2 in keys(solns[video])]
        # errp_video = [norm((R_dif*solns[video][key2].p + p_dif) - gt_all[video][key2][1][2]) for key2 in keys(solns[video])]
        errp_video = [norm(solns[video][key2].p - gt_all[video][key2][1][2]) for key2 in keys(solns[video])]


        # println((sum((errR_video .<= 5) .&& (errp_video .< 0.05))/length(errR_video)*100))
        append!(errR, errR_video)
        append!(errp, errp_video)
    end
    println("-----$method-----")
    # 2615 for mug, 2561 for camera
    @printf "5deg5cm: %.1f%%\n" (sum((errR .<= 5) .&& (errp .< 0.05))*100 / 2615)#length(errR))
    @printf "Rerr: %.1f deg (<0.1m)\n" (mean(errR[errp .< 0.1]))
    @printf "perr: %.1f cm  (<0.1m)\n" (mean(errp[errp .< 0.1])*100)
end