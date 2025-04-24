## Plot local solutions
# Run "local" first to get local variables
# Lorenzo Shaikewitz, 4/16/2025

# TODO load from serialized
# using Serialization

all_labels = ["SCF", "SCF-Obj", "Power", "G-N", "L-M", "Manopt", "SDP"]
all_errors = [errors_scf, errors_scfobj, errors_power, errors_gn, errors_lm, errors_manopt, errors_sdp]
all_times = [runtimes_scf, runtimes_scfobj, runtimes_power, runtimes_gn, runtimes_lm, runtimes_manopt, runtimes_sdp]
all_iters = [iter_scf, iter_scfobj, iter_power]

# Convert to DataFrame
using DataFrames, TexTables
# method, K, σm, rotation error, runtime
df = DataFrame(method=String[], K=Int[], σm=Float64[], errR=Float64[], time=Float64[], iters=Int[])
for i = 1:length(all_errors)
    for idxK = 1:length(Ks)
        for idxσ = 1:length(σms)
            l = length(all_errors[i][idxK][idxσ])
            if !(i in [1,2,3])
                iters = repeat([missing],l)
            else
                iters = all_iters[i][idxK][idxσ]
            end
            # row = [repeat([all_labels[i]],l) repeat([Ks[idxK]],l) repeat([σms[idxσ]],l) all_errors[i][idxK][idxσ] all_times[i][idxK][idxσ]]
            df_new = DataFrame(method=repeat([all_labels[i]],l), K=repeat([Ks[idxK]],l), σm=repeat([σms[idxσ]],l), 
                errR=Vector{Float64}(all_errors[i][idxK][idxσ]), time=Vector{Float64}(all_times[i][idxK][idxσ]), iters=iters)
            global df = vcat(df, df_new)
        end
    end
end

## TABLE OF RUN TIMES
# convert to ms
df.time *= 1000

# TABLE: for K = 4
names = []
tabs = []
for idxσ in [3,21]
    df_s = subset(df, :K => k -> k .== 4, :σm => s -> s .== σms[idxσ])
    tab = summarize_by(df_s, :method, [:time], stats=("Mean (ms)"=> x->mean(x), "p90 (ms)"=> x->quantile(x,.9)))
    push!(tabs, tab)
    push!(names, "$(σms[idxσ] / r)")
end
tab_times1 = join_table(Pair.(names,tabs)...)
# to_tex(tab_times1) |>print

# TABLE: for K = 25
names = []
tabs = []
for idxσ in [3,21]
    df_s = subset(df, :K => k -> k .== 25, :σm => s -> s .== σms[idxσ])
    tab = summarize_by(df_s, :method, [:time], stats=("Mean (ms)"=> x->mean(x), "p90 (ms)"=> x->quantile(x,.9)))
    push!(tabs, tab)
    push!(names, "$(σms[idxσ] / r)")
end
tab_times2 = join_table(Pair.(names,tabs)...)
# to_tex(tab_times2) |>print

tab_times_combined = join_table("K < N" => tab_times1, "K > N" => tab_times2)
# to_tex(tab_times_combined) |>print


## PLOT: NUMBER OF ITERATIONS
import Plots
using StatsPlots, Measures

iters = []
for i = 1:length(all_iters)
    push!(iters, [])
    for idxK = 1:length(Ks)
        summary_iters = zeros(3,length(σms))
        for idxσ = 1:length(σms)
            it = all_iters[i][idxK][idxσ]
            summary_iters[1,idxσ] = median(it)
            summary_iters[2,idxσ] = quantile(it,0.25)
            summary_iters[3,idxσ] = quantile(it,0.75)
        end
        push!(iters[i], summary_iters)
    end
end

p_iters1 = Plots.plot(ylabel = "Iterations", xlabel="Measurement Noise Scale")
for idx = 1:3
    Plots.plot!(σms ./ r, iters[idx][1][1,:], ribbon=(iters[idx][1][1,:] - iters[idx][1][2,:], iters[idx][1][3,:] - iters[idx][1][1,:]), label=all_labels[idx])
end

p_iters2 = Plots.plot(ylabel = "Iterations", xlabel="Measurement Noise Scale")
for idx = 1:3
    Plots.plot!(σms ./ r, iters[idx][2][1,:], ribbon=(iters[idx][2][1,:] - iters[idx][2][2,:], iters[idx][2][3,:] - iters[idx][2][1,:]), label=all_labels[idx])
end


## PLOT: SCALING WITH K



## PLOT: ROTATION ERROR vs. SDP
# preprocessing
errors = []
times = []
error_outliers = []
time_outliers = []
for i = 1:length(all_errors)
    errs = all_errors[i]
    ts = all_times[i]
    push!(errors, [])
    push!(times, [])
    push!(error_outliers, [])
    push!(time_outliers, [])
    for idxK = 1:length(Ks)
        push!(errors[i], [])
        push!(times[i], [])
        push!(error_outliers[i], [])
        push!(time_outliers[i], [])
        err = zeros(3,length(σms))
        t = zeros(3,length(σms))
        err_outliers = []
        t_outliers = []
        for idxσ = 1:length(σms)
            if length(errs[idxK][idxσ]) - 1 >= problems_per_config
                range = 2:problems_per_config
            else
                range = 1:length(errs[idxK][idxσ])
            end
            err[1,idxσ] = median(errs[idxK][idxσ][range])
            err[2,idxσ] = quantile(errs[idxK][idxσ][range],0.25)
            err[3,idxσ] = quantile(errs[idxK][idxσ][range],0.75)
            t[1,idxσ] = median(ts[idxK][idxσ][range])
            t[2,idxσ] = quantile(ts[idxK][idxσ][range], 0.25)
            t[3,idxσ] = quantile(ts[idxK][idxσ][range], 0.75)

            # get outliers too
            errs_this = errs[idxK][idxσ][range]
            push!(err_outliers, errs_this[errs_this .> err[3,idxσ]])
            push!(err_outliers, errs_this[errs_this .< err[2,idxσ]])
            ts_this = ts[idxK][idxσ][range]
            push!(t_outliers, ts_this[ts_this .> t[3,idxσ]])
            push!(t_outliers, ts_this[ts_this .< t[2,idxσ]])
        end
        println("$idxK, $i")
        errors[i][idxK] = err
        times[i][idxK] = t
        error_outliers[i][idxK] = err_outliers
        time_outliers[i][idxK] = t_outliers
    end
end

# K = 4
p_errors = Plots.plot(ylabel = "Rotation Error (deg)", xlabel="Measurement Noise Scale")
idx = 1
Plots.plot!(σms ./ r, errors[idx][1][1,:], ribbon=(errors[idx][1][1,:] - errors[idx][1][2,:], errors[idx][1][3,:] - errors[idx][1][1,:]), label=all_labels[idx])
idx = 7
Plots.plot!(σms ./ r, errors[idx][1][1,:], ribbon=(errors[idx][1][1,:] - errors[idx][1][2,:], errors[idx][1][3,:] - errors[idx][1][1,:]), label=all_labels[idx])

# K = 4, violin plot
plot_violin = Plots.plot(ylabel = "Rotation Error (deg)", xlabel="Measurement Noise Scale")
idx1 = 1
idx2 = 7
for idxσ in [3,7,13,21,41]
    σm = σms[idxσ]
    Plots.violin!(ones(1001)*σm/r,all_errors[idx1][1][idxσ],side=:left,label=false,color=1)
    Plots.violin!(ones(1001)*σm/r,all_errors[idx2][1][idxσ],side=:right,label=false,color=2)
end
plot_violin.series_list[1][:label] = all_labels[idx1]
plot_violin.series_list[3][:label] = all_labels[idx2]


# K = 25, violin plot
plot_violin2 = Plots.plot(ylabel = "Rotation Error (deg)", xlabel="Measurement Noise Scale")
idx1 = 1
idx2 = 7
for idxσ in [3,7,13,21,41]
    σm = σms[idxσ]
    Plots.violin!(ones(1001)*σm/r,all_errors[idx1][2][idxσ],side=:left,label=false,color=1)
    Plots.violin!(ones(1001)*σm/r,all_errors[idx2][2][idxσ],side=:right,label=false,color=2)
end
plot_violin2.series_list[1][:label] = all_labels[idx1]
plot_violin2.series_list[3][:label] = all_labels[idx2]


# K = 4, dot plot
plot_dot1 = Plots.plot(ylabel = "Rotation Error (deg)", xlabel="Measurement Noise Scale", dpi=300)
idx1 = 1
idx2 = 7
for idxσ in [3,7,13,21,41]
    σm = σms[idxσ]
    dotplot!(string.(round.(ones(1001)*σm/r,digits=2)),all_errors[idx1][1][idxσ],side=:left,label=false,mc=1,msw=0,ms=1.2)#marker=(1, stroke(0)))
    dotplot!(string.(round.(ones(1001)*σm/r,digits=2)),all_errors[idx2][1][idxσ],side=:right,label=false,mc=2,msw=0,ms=1.2)#marker=(2, stroke(0)))
end
plot_dot1.series_list[1][:label] = all_labels[idx1]
plot_dot1.series_list[2][:label] = all_labels[idx2]
Plots.plot!(dpi=300, fontfamily="Helvetica", grid=false, title="Comparison with Global Solver")

# K = 25, dot plot
plot_dot2 = Plots.plot(ylabel = "Rotation Error (deg)", xlabel="Measurement Noise Scale", dpi=300)
idx1 = 1
idx2 = 2
for idxσ in [3,7,13,21,41]
    σm = σms[idxσ]
    dotplot!(string.(round.(ones(1001)*σm/r,digits=2)),all_errors[idx1][1][idxσ],side=:left,label=false,mc=1,msw=0,ms=1.2)#marker=(1, stroke(0)))
    dotplot!(string.(round.(ones(1001)*σm/r,digits=2)),all_errors[idx2][1][idxσ],side=:right,label=false,mc=3,msw=0,ms=1.2)#marker=(2, stroke(0)))
end
plot_dot2.series_list[1][:label] = all_labels[idx1]
plot_dot2.series_list[2][:label] = all_labels[idx2]
Plots.plot!(dpi=300, fontfamily="Helvetica", grid=false,title="Objective Termination")

# plot both together
Plots.plot(plot_dot1,plot_dot2,size=(900,300), margin=8mm, framestyle=:box, legendcolumns=-1)
# savefig("combined.svg")