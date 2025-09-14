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
    methods = ["SCF", "GN"]
end
methods_to_run = []
if parsed_args["force"]
    methods_to_run = methods
else
    for m in methods
        if isfile("data/apolloscape/$m.dat")
            println("Using data in `data/apolloscape` for $m.")
        else
            push!(methods_to_run, m)
        end
    end
end

## load data
# load keypoint data
dets = JSON.parsefile("data/apolloscape/detections.json")

kpts_all = Dict()
gt_all = Dict()
for (car_name,kpts) in dets
    kpts_all[car_name] = Dict()
    gt_all[car_name] = Dict()
    for (i,d) in enumerate(kpts)
        kpts_all[car_name][i] = convert.(Float64, reduce(hcat,d["est_world_keypoints"]))' # [m]
        T = convert.(Float64,reduce(hcat,d["gt_pose"]))'
        gt_all[car_name][i] = (project2SO3(T[1:3,1:3]), T[1:3,4])
    end
end

# load CAD frame
shapes_dict = JSON.parsefile("data/apolloscape/shapes.json")
keys_shapes = collect(keys(shapes_dict))
shapes = cat([reduce(hcat,v)' for v in shapes_dict.(keys_shapes)]..., dims=3) # 3 x N x K

# create problem
# prob = FastPACE.Problem(size(shapes,2), size(shapes,3), 0.05, 0.2, shapes)


# solve!
if !isempty(methods_to_run)
    Random.seed!(0)
    λ = 0.5

    times_all = Dict(Pair.(methods_to_run, [Dict(Pair.(keys(kpts_all), [Dict() for k in keys(kpts_all)])) for i in methods_to_run]))
    gnc_success_all = Dict(Pair.(methods_to_run, [Dict(Pair.(keys(kpts_all), [Dict() for k in keys(kpts_all)])) for i in methods_to_run]))
    gnc_iters = Dict(Pair.(methods_to_run, [Dict(Pair.(keys(kpts_all), [Dict() for k in keys(kpts_all)])) for i in methods_to_run]))
    solns_all = Dict(Pair.(methods_to_run, [Dict(Pair.(keys(kpts_all), [Dict() for k in keys(kpts_all)])) for i in methods_to_run]))
    for car_name in collect(keys(kpts_all))
        println("Starting $car_name...")
        for frame in sort(collect(keys(kpts_all[car_name])))
            y = kpts_all[car_name][frame]

            # remove occluded points (0 depth)
            shapes_adj = shapes[:, (y[3,:] .!= 0) .&& (y[3,:] .!= Inf),:]
            y = y[:,(y[3,:] .!= 0) .&& (y[3,:] .!= Inf)]
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
                    out = @timed gnc(prob, y, λ, solvePACE_SCF; certify=false, cbar2 = 0.2, μUpdate = 1.4)
                    soln, inliers, success, iters = out.value
                elseif method == "SCFopt"
                    out = @timed gnc(prob, y, λ, solvePACE_SCF; certify=true, cbar2 = 0.15, μUpdate = 1.4)
                    soln, inliers, success, iters = out.value
                else
                    error("Method $method not implemented.")
                end

                gnc_success_all[method][car_name][frame] = success
                gnc_iters[method][car_name][frame] = iters
                time_dif = out.time - out.compile_time
                times_all[method][car_name][frame] = time_dif
                solns_all[method][car_name][frame] = soln
            end
        

            if mod(frame, 10) == 0
                print("$frame ")
            end
        end
        println("")
    end

    # save
    for method in methods_to_run
        serialize("data/apolloscape/$method.dat", Dict("times"=>times_all[method], "gnc_success"=>gnc_success_all[method], 
                                            "solns"=>solns_all[method], "gnc_iters"=>gnc_iters[method]))

        # save just solns in JSON format (TODO fix for this JSON)
        # solns_json = Dict()
        # for key in keys(solns[method])
        #     key2 = split(key,".")[1]
        #     solns_json[key2] = Dict{Int,Any}()
        #     for frame in keys(solns[method][key])
        #         solns_json[key2][frame] = Dict("p"=>vec(solns[method][key][frame].p), 
        #             "R"=>vec(solns[method][key][frame].R), "c"=>vec(solns[method][key][frame].c))
        #     end
        # end
        # open("data/apolloscape/$method.json","w") do f
        #     JSON.print(f, solns_json)
        # end
    end
    println("")
end

# Next: visualize
for (i,method) in enumerate(methods)
    data = deserialize("data/apolloscape/$method.dat")
    solns = data["solns"]

    errR = [[roterror(solns[car_name][frame].R, gt_all[car_name][frame][1]) for frame in keys(solns[car_name])] for car_name in keys(solns)]
    errR = reduce(vcat, errR)
    errp = [[norm(solns[car_name][frame].p - gt_all[car_name][frame][2]) for frame in keys(solns[car_name])] for car_name in keys(solns)]
    errp = reduce(vcat, errp)
    errc = [[keys_shapes[argmax(solns[car_name][frame].c)] == car_name for frame in keys(solns[car_name])] for car_name in keys(solns)]
    errc = reduce(vcat, errc)

    println("-----$method-----")
    println("R error: $(mean(errR)) deg")
    println("p error: $(mean(errp)) m")
    println("c error: $(sum(errc) / length(errc) * 100) %")
end