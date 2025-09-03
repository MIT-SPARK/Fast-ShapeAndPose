## Run on CAST dataset with GNC.
# Measure:
# - pose estimate
# - total runtime
# - GNC iterations
# - optimality certificate
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
    methods = ["SCF", "GN"]
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


# solve!
if !isempty(methods_to_run)
    λ = 0.

    times = Dict(Pair.(methods_to_run, [Dict() for i in methods_to_run]))
    gnc_success = Dict(Pair.(methods_to_run, [Dict() for i in methods_to_run]))
    solns = Dict(Pair.(methods_to_run, [Dict() for i in methods_to_run]))
    for frame in sort(collect(keys(kpts_test)))
        y = kpts_test[frame][1]

        for method in methods_to_run
            if method == "SCF"
                out = @timed gnc(prob, y, λ, solvePACE_SCF; cbar2 = 0.005, μUpdate = 1.4) # 0.005 is good
                soln, inliers, success = out.value
            elseif method == "GN"
                out = @timed gnc(prob, y, λ, solvePACE_GN; cbar2 = 0.005, μUpdate = 1.4) # 0.005 is good
                soln, inliers, success = out.value
            else
                error("Method $method not implemented.")
            end

            gnc_success[method][frame] = success
            time_dif = out.time - out.compile_time
            times[method][frame] = time_dif
            solns[method][frame] = soln
        end
       

        if mod(frame, 10) == 0
            print("$frame ")
        end
    end

    # save
    for method in methods_to_run
        serialize("data/cast/$method.dat", Dict("times"=>times[method], "gnc_success"=>gnc_success[method], "solns"=>solns[method]))
    end
end

# visualize!
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

# TODO: time histogram