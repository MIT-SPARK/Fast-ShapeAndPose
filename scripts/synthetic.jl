## Run against other local solvers on synthetic data
# Solvers:
# - SDP
# - Manopt
# - GN
# - LM
# - Power iteration?
# - Objective termination?
# Measure:
# - estimation error (rotation only)
# - runtime
# - number of iterations (if relevant)
# - optimality certificate (if relevant)
#
# I recommend running one method at a time to keep Julia from optimizing.
# Lorenzo Shaikewitz, 8/28/2025

using LinearAlgebra
using ArgParse
using Serialization

using SimpleRotations
using FastPACE

## Command-line arguments
s = ArgParseSettings()
@add_arg_table s begin
    "--force"
        help = "Rerun even if data already saved"
        action = :store_true
    "method"
        help = "method to run: SCF, GN, etc. or all. Can call multiple like [SCF,GN] (no spaces)"
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
        if isfile("data/synthetic/$m.dat")
            println("Using data in `data/synthetic` for $m.")
        else
            push!(methods_to_run, m)
        end
    end
end

# generate data

# parameters
r = 0.2
N = 10
Ks = [4,25]
σms = round.([0.25, 0.75, 1.5, 2.5, 5.0] .* r, digits=4)
problems_per_config = 10000

# generate all data (generate once, load from file afterwards)
# data = gendata(r, N, Ks, σms, problems_per_config)
# q0 = normalize(randn(4))
# serialize("data/synthetic/problem_data.dat", (data,q0))
# error("Data generated!")

# load data
data, q0 = deserialize("data/synthetic/problem_data.dat")
R0 = quat2rotm(q0)

# solve!
if !isempty(methods_to_run)
    errors = Dict()
    runtimes = Dict()
    certs = Dict()

    for method in methods_to_run
        status = nothing
        if method == "SDP"
            e, time, gaps, status = runlocaliter(data, solvePACE_SDP)
        elseif method == "Manopt"
            e, time = runlocal(data, solvePACE_Manopt; R0=R0)
        elseif method == "GN"
            e, time = runlocal(data, solvePACE_GN; max_iters = 250, R₀=R0)
        elseif method == "LM"
            e, time = runlocal(data, solvePACE_GN; λ_lm=0.1, max_iters = 250, R₀=R0)
        elseif method == "SCF"
            e, time, iter, status = runlocaliter(data, solvePACE_SCF; certify=false, max_iters = 100, q0=q0)
        elseif method == "SCFopt"
            e, time, iter, status = runlocaliter(data, solvePACE_SCF; certify=true, max_iters = 100, q0=q0)
        else
            error("Method $method not implemented.")
        end

        errors[method] = e
        runtimes[method] = time
        certs[method] = status
    end

    # save
    for method in methods_to_run
        serialize("data/synthetic/$method.dat", Dict("errors"=>errors[method], "times"=>runtimes[method], "certs"=>certs[method]))
    end
end

# visualize
using DataFrames, TexTables
using Statistics

# load data
errors = Dict()
times = Dict()
certs = Dict()
for method in methods
    d = deserialize("data/synthetic/$method.dat")
    errors[method] = d["errors"]
    times[method] = d["times"]
    certs[method] = d["certs"]
end


df = DataFrame(method=String[], K=Int[], σm=Float64[], errR=Float64[], time=Float64[], certs=[])
for method in methods
    for idxK = 1:length(Ks)
        for idxσ = 1:length(σms)
            l = length(errors[method][idxK][idxσ])
            if isnothing(certs[method])
                c = repeat([FastPACE.LOCAL_SOLUTION],l)
            else
                c = certs[method][idxK][idxσ]
            end
            df_new = DataFrame(method=repeat([method],l), K=repeat([Ks[idxK]],l), σm=repeat([σms[idxσ]],l), 
                errR=Vector{Float64}(errors[method][idxK][idxσ]), time=Vector{Float64}(times[method][idxK][idxσ]), certs=c)
            global df = vcat(df, df_new)
        end
    end
end

using Plots, StatsPlots
if parsed_args["method"] == "all"
    # rotation error for SDP vs SCF vs GN
    # K = 4, boxlot
    methods_to_plot = ["GN", "SCF", "SDP"]

    boxdata = []
    mdata = []
    σdata = []
    for (idxσ,σm) in enumerate(σms)
        for m in methods_to_plot
            n = length(errors[m][1][idxσ])
            append!(boxdata, errors[m][1][idxσ]) # rotation errors
            append!(mdata, repeat([m], n))
            append!(σdata, repeat([round((σm / r),digits=2)], n))
        end
        
        # boxplot of positive certificates (SCF only)
        df_SCFcertified = subset(df, :K => k -> k .== 4, :σm => s -> s .== σm, :method => m -> m .== "SCFopt", :certs => c -> c .== FastPACE.GLOBAL_CERTIFIED)
        e = df_SCFcertified[:,:errR]
        n = length(e)
        append!(boxdata, e) # rotation errors
        append!(mdata, repeat(["SCF2"], n))
        append!(σdata, repeat([round((σm / r),digits=2)], n))
    end
    plot_box = groupedboxplot(σdata, boxdata; group=mdata, msw=0., fillalpha=0.75, linecolor=["#007ecc" "#d84c1f" "#8b0000" "#338740"], fillcolor=[1 2 "#8b0001" 3], markercolor=[1 2 "#8b0000" 3],ms=1.5)
    # Plots.plot!(ylabel = "Rotation Error (log deg)", xlabel="Measurement Noise Scale", yscale=:log10)
    Plots.plot!(yscale=:log10)
    Plots.plot!(dpi=300, fontfamily="Helvetica")
    Plots.plot!(thickness_scaling=1.5)

    
    
    # certificate percentages (SDP vs SCFopt)
    println("Certificate:")
    for idxσ in 1:5#[1,4]
        df_SCFopt = subset(df, :K => k -> k .== 4, :σm => s -> s .== σms[idxσ], :method => m -> m .== "SCFopt")
        certified_SCFopt = sum(df_SCFopt[:,:certs] .== FastPACE.GLOBAL_CERTIFIED)
        df_SDP = subset(df, :K => k -> k .== 4, :σm => s -> s .== σms[idxσ], :method => m -> m .== "SDP")
        certified_SDP = sum(df_SDP[:,:certs] .== FastPACE.GLOBAL_CERTIFIED)
        println("[K=4, σm=$(round((σms[idxσ] / r),digits=2))]  SCF*: $certified_SCFopt, SDP: $certified_SDP (/ $(size(df_SCFopt,1)), $(size(df_SDP,1)))")
    end
    println("")
end

## TABLE OF RUN TIMES
# convert to ms
df.time *= 1000

# TABLE: for K = 4
names = []
tabs = []
for idxσ in [1,4]
    df_s = subset(df, :K => k -> k .== 25, :σm => s -> s .== σms[idxσ])
    tab = summarize_by(df_s, :method, [:time], stats=("Mean (ms)"=> x->mean(x), "p90 (ms)"=> x->quantile(x,.9)))
    push!(tabs, tab)
    push!(names, "$(round((σms[idxσ] / r),digits=2))")
end
tab_times2 = join_table(Pair.(names,tabs)...)

# TABLE: for K = 4
names = []
tabs = []
for idxσ in [1,4]
    df_s = subset(df, :K => k -> k .== 4, :σm => s -> s .== σms[idxσ])
    tab = summarize_by(df_s, :method, [:time], stats=("Mean (ms)"=> x->mean(x), "p90 (ms)"=> x->quantile(x,.9)))
    push!(tabs, tab)
    push!(names, "$(round((σms[idxσ] / r),digits=2))")
end
tab_times1 = join_table(Pair.(names,tabs)...)
# to_tex(tab_times1) |>print