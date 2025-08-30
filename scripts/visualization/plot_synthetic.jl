all_labels = ["SCF", "SCF-Cert", "G-N", "L-M"]
all_errors = [errors_scf, errors_scfcert, errors_gn, errors_lm]
all_times = [runtimes_scf, runtimes_scfcert, runtimes_gn, runtimes_lm]
all_iters = [iter_scf, iter_scfcert]

# Convert to DataFrame
using DataFrames, TexTables
using Statistics
# method, K, σm, rotation error, runtime
df = DataFrame(method=String[], K=Int[], σm=Float64[], errR=Float64[], time=Float64[], iters=Int[])
for i = 1:length(all_errors)
    for idxK = 1:length(Ks)
        for idxσ = 1:length(σms)
            l = length(all_errors[i][idxK][idxσ])
            if !(i in [1,2])
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
for idxσ in [1,2]
    df_s = subset(df, :K => k -> k .== 4, :σm => s -> s .== σms[idxσ])
    tab = summarize_by(df_s, :method, [:time], stats=("Mean (ms)"=> x->mean(x), "p90 (ms)"=> x->quantile(x,.9)))
    push!(tabs, tab)
    push!(names, "$(σms[idxσ] / r)")
end
tab_times1 = join_table(Pair.(names,tabs)...)
# to_tex(tab_times1) |>print