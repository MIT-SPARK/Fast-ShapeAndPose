# generate data
function gendata(r, N, Ks, σms, problems_per_config)
    data = []
    for K in Ks
        data_K = []
        for σm in σms
            data_σm = []
            for i = 1:problems_per_config
                prob, gt, y = genproblem(N=N, K=K, σm=σm, r=r)
                lam = 0.0
                if K >= N
                    lam = 1.0
                end
                weights = ones(prob.N)

                d = [prob, gt, y, weights, lam]
                push!(data_σm, d)
            end
            push!(data_K, data_σm)
        end
        push!(data, data_K)
    end
    return data
end

# run a single local solver
function runlocal(data, solver; kwargs...)
    errors = []
    runtimes = []
    for data_K in data
        errors_K = []; runtimes_K = []
        for data_σm in data_K
            print("$(data_σm[1][1].σm) ")
            errors_σm = []; runtimes_σm = []
            for (prob, gt, y, weights, lam) in data_σm
                try
                out = @timed solver(prob, y, weights, lam; kwargs...)
                soln = out.value[1]
                _, R_err = rotm2axang(gt.R'*soln.R); R_err *= 180/π
                if out.compile_time == 0.0
                    push!(errors_σm, R_err)
                    push!(runtimes_σm, out.time)
                else
                    println("Compiling...")
                end
                catch
                    println("Error!")
                end
            end
            push!(errors_K, errors_σm)
            push!(runtimes_K, runtimes_σm)
        end
        push!(errors, errors_K)
        push!(runtimes, runtimes_K)
    end
    return errors, runtimes
end

# run a single local solver
function runlocaliter(data, solver; kwargs...)
    errors = []
    runtimes = []
    iters = []
    for data_K in data
        errors_K = []; runtimes_K = []; iters_K = []
        for data_σm in data_K
            print("$(data_σm[1][1].σm) ")
            errors_σm = []; runtimes_σm = []; iters_σm = []
            for (prob, gt, y, weights, lam) in data_σm
                out = @timed solver(prob, y, weights, lam; kwargs...)
                soln = out.value[1]
                _, R_err = rotm2axang(gt.R'*soln.R); R_err *= 180/π
                if out.compile_time == 0.0
                    push!(errors_σm, R_err)
                    push!(runtimes_σm, out.time)
                    push!(iters_σm, out.value[end])
                else
                    println("Compiling...")
                end
            end
            push!(errors_K, errors_σm)
            push!(runtimes_K, runtimes_σm)
            push!(iters_K, iters_σm)
        end
        push!(errors, errors_K)
        push!(runtimes, runtimes_K)
        push!(iters, iters_K)
    end
    return errors, runtimes, iters
end