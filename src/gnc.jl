# implements Graduated Non-Convexity with TLS cost
# https://arxiv.org/pdf/1909.08605

"""
    gnc(prob, y, solver [; maxiterations=100, stopthresh=1e-6, μUpdate=1.4, cbar2=1.0])

Run GNC given problem data `prob`, `y`, and solver function `soln, cost, residuals = solver(prob, y, weights, λ)`
where `residuals` are squared residuals for each input, size `prob.N`.

Runs up to cost diff `stopthresh` or `maxiterations`. Update `μ->μUpdate*μ`.
Use outlier rejection threshold `cbar2`.
"""
function gnc(prob, y, λ, solver; maxiterations=100, stopthresh=1e-6, μUpdate=1.4, cbar2=1.0, debug=false, solverargs=Dict())
    weights = ones(prob.N)
    last_weights = weights
    last_cost = 1e6
    Δcost = 1e6
    soln = nothing
    success = false
    μ = 0.05

    for iter in 1:maxiterations
        # termination conditions
        if Δcost < stopthresh
            if debug
                @printf "GNC converged %3.2e < %3.2e.\n" Δcost stopthresh
            end
            success = true
            break
        end
        if maximum(abs.(weights)) < 1e-6
            if debug
                printstyled("GNC encounters numerical issues.\n", color=:red)
            end
            break
        end

        # non-minimal solver
        soln, cost, residuals = solver(prob, y, weights, λ; solverargs...)
        # add constant cost for each outlier
        cost += cbar2*sum(weights.==0)

        # initialize μ
        if iter == 1
            μ = cbar2 / (2*maximum(residuals) - cbar2) # Remark 5
        end

        # weights update (eq. 14)
        last_weights = weights
        weights[residuals .<= [μ/(μ+1)*cbar2]] .= 1.
        weights[residuals .> [μ/(μ+1)*cbar2]] = sqrt.(cbar2./residuals[residuals .> μ/(μ+1)*cbar2] * μ*(μ+1)) .- μ
        weights[residuals .>= [(μ+1)/μ*cbar2]] .= 0.
        # weights = weights_update(residuals, μ, cbar2)

        # cost difference
        Δcost = abs(cost - last_cost)
        last_cost = cost

        # update μ
        μ = μ*μUpdate

        if debug
            @printf "%3d | cost: %.2e, weights: %.2f\n" iter cost sum(last_weights)
            # println(residuals)
            # println(weights)
        end
    end

    # return inliers + solution
    inliers = collect(1:prob.N)[last_weights .> 1e-6]
    return soln, inliers, success

end



function weights_update(residuals, μ, cbar2)
    N = length(residuals)
    model = Model(Mosek.Optimizer)
    @variable(model, w[1:N] .>= 0)
    @constraint(model, w .<= 1)

    # constrain to one assocation per measurement
    for i = 1:10
        @constraint(model, sum(w[10*(i-1)+1:10*i]) == 1)
    end
    # constrain to one association per shape point
    for i = 1:10
        @constraint(model, sum(w[i:10:end]) <= 1)
    end

    @objective(model, Min, sum([w[i]*residuals[i]*(μ + w[i]) + μ*(1 - w[i])*cbar2 for i = 1:N]))
    set_silent(model)
    optimize!(model)
    weights = abs.(value.(w))
    weights[weights .<= 1e-4] .= 0.
    return weights
end