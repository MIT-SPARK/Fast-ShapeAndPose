## Run on CAST dataset with GNC.
# Measure:
# - pose estimate
# - total runtime
# - GNC iterations
# - optimality certificate (?)
# Lorenzo Shaikewitz, 8/29/2025

using LinearAlgebra
using Statistics
import JSON
using MAT
using ArgParse
using Serialization
using Printf
import Plots
using StatsPlots
using Random

using SimpleRotations

using FastPACE

## Command-line arguments
s = ArgParseSettings()
@add_arg_table s begin
    "--force"
        help = "Rerun even if data already saved"
        action = :store_true
    "method"
        help = "method to run: SCF, GN, or all. Can call multiple like [SCF,GN] (no spaces)"
        default = "all"
end
parsed_args = parse_args(ARGS, s)

# don't re-run methods unless force has been called
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
        if isfile("data/cast/$m.dat")
            println("Using data in `data/cast` for $m.")
        else
            push!(methods_to_run, m)
        end
    end
end

# load keypoint data
dets = JSON.parsefile("data/cast/cast.json")
camK = convert.(Float64,reduce(hcat,dets[1]["K_mat"])')

kpts_test = Dict()
gt_test = Dict()
for (i,d) in enumerate(dets)
    kpts_test[i] = Dict(1=>convert.(Float64,reduce(hcat,d["est_world_keypoints"])) ./ 1000.)
    T = convert.(Float64,reduce(hcat,d["gt_teaser_pose"]))'
    gt_test[i] = Dict(1=>(T[1:3,1:3], T[1:3,4]/1000.))
end

# load CAD frame
file = matopen("data/cast/racecar_lib.mat")
shapes = read(file, "shapes") # 3 x N x K
close(file)

# create problem
prob = FastPACE.Problem(size(shapes,2), size(shapes,3), 0.05, 0.2, shapes)


# solve!
if !isempty(methods_to_run)
    Random.seed!(0)
    λ = 0.

    times_all = Dict(Pair.(methods_to_run, [Dict() for i in methods_to_run]))
    gnc_success_all = Dict(Pair.(methods_to_run, [Dict() for i in methods_to_run]))
    gnc_iters = Dict(Pair.(methods_to_run, [Dict() for i in methods_to_run]))
    solns_all = Dict(Pair.(methods_to_run, [Dict() for i in methods_to_run]))
    for frame in sort(collect(keys(kpts_test)))
        y = kpts_test[frame][1]

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

            gnc_success_all[method][frame] = success
            gnc_iters[method][frame] = iters
            time_dif = out.time - out.compile_time
            times_all[method][frame] = time_dif
            solns_all[method][frame] = soln
        end
       

        if mod(frame, 10) == 0
            print("$frame ")
        end
    end

    # save
    for method in methods_to_run
        serialize("data/cast/$method.dat", Dict("times"=>times_all[method], "gnc_success"=>gnc_success_all[method], 
                                            "solns"=>solns_all[method], "gnc_iters"=>gnc_iters[method]))
    end
    println("")
end


# load data
using DataFrames, TexTables
using Statistics
df = DataFrame(method=String[], frame=Int[], errR=Float64[], errp=Float64[], time=Float64[], success=Bool[], gnc_iters=Int[])
for method in methods
    data = deserialize("data/cast/$method.dat")
    solns = data["solns"]
    frames = Int.(sort(collect(keys(solns))))


    errR = [roterror(solns[frame].R, gt_test[frame][1][1]) for frame in frames]
    errp = [norm(solns[frame].p - gt_test[frame][1][2]) for frame in frames]

    df_new = DataFrame(method=repeat([method],length(frames)), frame=frames, 
                        time=data["times"].(frames), success=data["gnc_success"].(frames), 
                        errR=errR, errp=errp, gnc_iters=data["gnc_iters"].(frames))
    global df = vcat(df, df_new)
end

# table of results
# convert to ms
df.time *= 1000
df_s = subset(df, :success => g -> g.==true)
tab_cast = summarize_by(df_s, :method, [:time, :errR, :gnc_iters], stats=("Mean"=> x->mean(x), "p90"=> x->quantile(x,.9)))
# to_tex(tab_cast) |>print
display(tab_cast)

# print stats and visualize
time_plot = Plots.plot()
for (i,method) in enumerate(methods)
    data = deserialize("data/cast/$method.dat")
    times = data["times"]
    gnc_success = data["gnc_success"]
    solns = data["solns"]
    frames = keys(solns)

    errR = [roterror(solns[frame].R, gt_test[frame][1][1]) for frame in frames]
    errp = [norm(solns[frame].p - gt_test[frame][1][2]) for frame in frames]

    println("------$method------")
    
    @printf "R error: %.1f°, t error: %.1f mm (%d frames)\n" mean(errR) mean(errp)*1000 length(frames)
    @printf "R error: %.1f°, t error: %.1f mm (%d successful)\n" mean(errR[gnc_success.(frames)]) mean(errp[gnc_success.(frames)])*1000 sum(gnc_success.(frames))
    # Plots.violin!(ones(length(frames))*i,times.(frames),label=method)
    Plots.boxplot!(repeat([method],length(frames)), times.(frames), label=false)#, side=(i == 1) ? :left : :right)
end
Plots.plot!(ylims=(0,0.02), ylabel="Time (s)")
time_plot

# TODO: table with columns: mean, p90, Rerr (mean)