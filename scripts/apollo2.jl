## Run on ApolloCar3D dataset
#
# Lorenzo Shaikeiwitz, 9/13/2025

using ArgParse
using LinearAlgebra
using Statistics
using Serialization
using Glob
import JSON
using MAT
using Printf
import Plots
using Random
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
end
parsed_args = parse_args(ARGS, s)
# parsed_args["force"] = true

# don't re-run methods unless force has been called
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
        if isfile("data/apolloscape2/$m.dat")
            println("Using data in `data/apolloscape2` for $m.")
        else
            push!(methods_to_run, m)
        end
    end
end

## load data
# load keypoint data
# dets = JSON.parsefile("data/apolloscape2/detections.json")
dets = JSON.parsefile("data/apolloscape2/robin_detections.json")

kpts_all = Dict()
gt_all = Dict()
robin_all = Dict()
for (img_name,dets_img) in dets
    kpts_all[img_name] = []
    gt_all[img_name] = []
    robin_all[img_name] = []
    for d in dets_img
        push!(kpts_all[img_name], convert.(Float64, reduce(hcat,d["est_world_keypoints"]))') # [m]
        T = convert.(Float64,reduce(hcat,d["gt_pose"]))'
        gt_shape = Int(d["car_id"])
        push!(gt_all[img_name], (project2SO3(T[1:3,1:3]), T[1:3,4], gt_shape))
        # robin
        push!(robin_all[img_name], Dict("time"=>d["robin_time"], "inliers"=>d["robin_inliers"]))
    end
end

# load id to name dict
id2name = JSON.parsefile("data/apolloscape2/car_ids.json")
ids = sort(parse.(Int,collect(keys(id2name))))
keys_shapes_car_names = id2name.(string.(ids))

# load CAD frame
shapes_dict = JSON.parsefile("data/apolloscape2/shapes.json")
keys_shapes = keys_shapes_car_names
shapes = cat([reduce(hcat,v)' for v in shapes_dict.(keys_shapes)]..., dims=3) # 3 x N x K

# create problem
# prob = FastPACE.Problem(size(shapes,2), size(shapes,3), 0.05, 0.2, shapes)


# solve!
if !isempty(methods_to_run)
    Random.seed!(0)
    # λ = 1500.
    λ = 15000.

    img_names = collect(keys(kpts_all))

    times_all = Dict(Pair.(methods_to_run, [Dict(Pair.(img_names, [Dict() for k in img_names])) for i in methods_to_run]))
    gnc_success_all = Dict(Pair.(methods_to_run, [Dict(Pair.(img_names, [Dict() for k in img_names])) for i in methods_to_run]))
    gnc_iters = Dict(Pair.(methods_to_run, [Dict(Pair.(img_names, [Dict() for k in img_names])) for i in methods_to_run]))
    solns_all = Dict(Pair.(methods_to_run, [Dict(Pair.(img_names, [Dict() for k in img_names])) for i in methods_to_run]))
    print("Running $(length(img_names)) frames...")
    for (i, img_name) in enumerate(img_names)
        for (car_num, y) in enumerate(kpts_all[img_name])

            # remove robin points
            points_filter = (y[3,:] .!= 0) .&& (y[3,:] .!= Inf)
            use_robin = false
            if length(robin_all[img_name][car_num]["inliers"]) > 0
                robin_filter = zeros(Bool, size(y,2))
                robin_filter[robin_all[img_name][car_num]["inliers"] .+ 1] .= true
                if sum(points_filter .&& robin_filter) >= 3
                    # actually use robin
                    points_filter = points_filter .&& robin_filter
                    use_robin = true
                else
                    # Main.@infiltrate
                end
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
                    out = @timed gnc(prob, y, λ, solvePACE_SDP; cbar2 = 0.15, μUpdate = 1.4)
                    soln, inliers, success, iters = out.value
                elseif method == "Manopt"
                    out = @timed gnc(prob, y, λ, solvePACE_Manopt; cbar2 = 0.15, μUpdate = 1.4)
                    soln, inliers, success, iters = out.value
                elseif method == "GN"
                    out = @timed gnc(prob, y, λ, solvePACE_GN; cbar2 = 0.15, μUpdate = 1.4)
                    soln, inliers, success, iters = out.value
                elseif method == "LM"
                    out = @timed gnc(prob, y, λ, solvePACE_GN; λ_lm=0.1, cbar2 = 0.15, μUpdate = 1.4)
                    soln, inliers, success, iters = out.value
                elseif method == "SCF"
                    out = @timed gnc(prob, y, λ, solvePACE_SCF; certify=false, cbar2 = 0.15, μUpdate = 1.4, max_iters=100)
                    soln, inliers, success, iters = out.value
                elseif method == "SCFopt"
                    out = @timed gnc(prob, y, λ, solvePACE_SCF; certify=true, cbar2 = 0.15, μUpdate = 1.4, max_iters=250)
                    soln, inliers, success, iters = out.value
                else
                    error("Method $method not implemented.")
                end

                gnc_success_all[method][img_name][car_num] = success
                gnc_iters[method][img_name][car_num] = iters
                time_dif = out.time - out.compile_time
                # add ROBIN time
                if use_robin
                    time_dif += robin_all[img_name][car_num]["time"]
                end
                times_all[method][img_name][car_num] = time_dif
                solns_all[method][img_name][car_num] = soln
            end
        end
        
        if mod(i, 10) == 0
            print("$i ")
        end
    end

    # save
    for method in methods_to_run
        serialize("data/apolloscape2/$method.dat", Dict("times"=>times_all[method], "gnc_success"=>gnc_success_all[method], 
                                            "solns"=>solns_all[method], "gnc_iters"=>gnc_iters[method]))

        # save just solns in JSON format (TODO fix for this JSON)
        solns = solns_all[method]
        solns_json = Dict()
        for img_name in keys(solns)
            solns_json[img_name] = []
            for car_num in keys(solns[img_name])
                push!(solns_json[img_name], Dict("p"=>vec(solns[img_name][car_num].p), 
                    "R"=>vec(solns[img_name][car_num].R), "c"=>vec(solns[img_name][car_num].c),
                    "gt_car_name" => gt_all[img_name][car_num][3]))
            end
        end
        open("data/apolloscape2/$method.json","w") do f
            JSON.print(f, solns_json)
        end
    end
    println("")
end

# Next: visualize
for (i,method) in enumerate(methods)
    data = deserialize("data/apolloscape2/$method.dat")
    solns = data["solns"]

    errR = [[roterror(solns[img_name][car_num].R, gt_all[img_name][car_num][1]) for car_num in 1:length(solns[img_name])] for img_name in keys(solns)]
    errR = reduce(vcat, errR)
    errp = [[norm(solns[img_name][car_num].p - gt_all[img_name][car_num][2]) for car_num in 1:length(solns[img_name])] for img_name in keys(solns)]
    errp = reduce(vcat, errp)
    errc = [[argmax(solns[img_name][car_num].c)[1]-1 == gt_all[img_name][car_num][3] for car_num in 1:length(solns[img_name])] for img_name in keys(solns)]
    errc = reduce(vcat, errc)
    Main.@infiltrate

    times_dict = data["times"]
    times = reduce(vcat,collect.(values.(values(times_dict))))

    gnc_dict = data["gnc_iters"]
    gnc_iters = reduce(vcat,collect.(values.(values(gnc_dict))))

    println("-----$method-----")
    @printf "Rerr: %.1f deg\n" (mean(errR))
    @printf "perr: %.1f cm\n" (mean(errp)*100)
    println("c error: $(sum(errc) / length(errc) * 100) %")
    @printf "GNCi: %.1f\n" (mean(gnc_iters))
    @printf "Time: %.2f ms\n" (mean(times)*1000)
    # println("R error: $(mean(errR)) deg")
    # println("p error: $(mean(errp)) m")
    # println("c error: $(sum(errc) / length(errc) * 100) %")
    # println("GNC iters: $(mean(gnc_iters))")
    # println("Time: $(mean(times))")
end