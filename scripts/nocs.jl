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
    methods = ["SCFopt", "SCF", "GN", "LM", "SDP", "Manopt"]
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

        for frame in sort(collect(keys(kpts_test)))
            y = kpts_test[frame][1]
            
            # remove occluded points (0 depth)
            shapes_adj = shapes[:, y[3,:] .!= 0,:]
            y = y[:,y[3,:] .!= 0]
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

                time_dif += 

                gnc_success[method][json][frame] = success
                gnc_iters[method][json][frame] = iters
                time_dif = out.time - out.compile_time
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

# visualize!
time_plot = Plots.plot()
jsons = sort(collect(keys(kpts_all)))
for (i,method) in enumerate(methods)
    data = deserialize("data/nocs/$(parsed_args["object"])/$method.dat")
    solns = data["solns"]

    errR = [[roterror(solns[key1][key2].R, gt_all[key1][key2][1][1]) for key2 in keys(solns[key1])] for key1 in keys(solns)]
    errR = reduce(vcat, errR)
    errp = [[norm(solns[key1][key2].p - gt_all[key1][key2][1][2]) for key2 in keys(solns[key1])] for key1 in keys(solns)]
    errp = reduce(vcat, errp)

    times_dict = data["times"]
    times = reduce(vcat,collect.(values.(values(times_dict))))

    gnc_dict = data["gnc_iters"]
    gnc_iters = reduce(vcat,collect.(values.(values(gnc_dict))))

    println("-----$method-----")
    println("5deg5cm: $(sum((errR .<= 5) .&& (errp .< 0.05))/length(errR)*100)")
    println("R error: $(mean(errR[errp .< 0.1])) deg (<0.1m)")
    println("p error: $(mean(errp[errp .< 0.1])) m (<0.1m)")
    println("GNC iters: $(mean(gnc_iters))")
    println("Time: $(mean(times))")

    # for json in jsons

        



    # for (jsonnum,json) in enumerate(jsons)
    #     println("\n-----$json ($(length(keys(data["times"][json]))) frames)-----")
    #     times = data["times"][json]
    #     gnc_success = data["gnc_success"][json]
    #     solns = data["solns"][json]
    #     frames = keys(solns)

    #     if normalize_frames[jsonnum] > 0
    #         normalize_frame = normalize_frames[jsonnum]
    #         gt_norm  = [project2SO3(gt_all[json][normalize_frame][1][1])  gt_all[json][normalize_frame][1][2]; 0 0 0 1]
    #         est_norm = [solns[normalize_frame].R  solns[normalize_frame].p; 0 0 0 1]
    #         diff = est_norm*inv(gt_norm)
            
    #         errR = zeros(length(frames))
    #         errp = zeros(length(frames))
    #         for (i,frame) in enumerate(frames)
    #             T_diff = diff*[gt_all[json][frame][1][1]  gt_all[json][frame][1][2]; 0 0 0 1]
    #             gtR = T_diff[1:3,1:3]
    #             gtp = T_diff[1:3,4]
                
    #             errR[i] = roterror(solns[frame].R, gtR)
    #             errp[i] = norm(solns[frame].p - gtp)
    #         end
    #     else
    #         errR = [roterror(solns[frame].R, project2SO3(gt_all[json][frame][1][1])) for frame in frames]
    #         errp = [norm(solns[frame].p - gt_all[json][frame][1][2]) for frame in frames]
    #     end


    #     println("$method:")
    #     @printf "R error: %.1f°, t error: %.1f mm\n" mean(errR) mean(errp)*1000
    #     @printf "R error: %.1f°, t error: %.1f mm (%d successful frames)\n" mean(errR[gnc_success.(frames)]) mean(errp[gnc_success.(frames)])*1000 sum(gnc_success.(frames))
    #     # these errors are going to be large: we estimate shape & pose, 
    #     # so pose error is not a great metric! see certifiable tracking for a little bit better version
        
    #     # Plots.violin!(repeat([jsonnum],length(frames)),times.(frames), side=(i == 1) ? :left : :right, c=i, label=(jsonnum==1) ? method : false)
    #     Plots.boxplot!(repeat([jsonnum+ 0.2*(i-1)],length(frames)), times.(frames), label=(jsonnum==1) ? method : false, c=i, notch=true)
    # end
end
# Plots.plot!(ylabel="Time (s)")#, ylims=(0,0.2))
# Plots.plot!(ylims=(0,0.05))
# time_plot