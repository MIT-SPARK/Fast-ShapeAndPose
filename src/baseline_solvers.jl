## Baseline solvers for category-level shape and pose estimation
# - SDP (standard Shor relaxation, global optimality)
# - Manopt (local on-manifold solver)
# - Gauss-Newton (traditional local solver)
# Lorenzo Shaikewitz, 8/29/2025


function constraintsRHR()
    q_eqs = []
    # r1 x r2 - r3
    H = zeros(9,9)
    H[2,6] = 1; H[3,5] = -1; H += H'
    c = zeros(9); c[7] = -1.
    Q = [H  c; c' 0.]
    push!(q_eqs, Q)
    H = zeros(9,9)
    H[3,4] = 1; H[1,6] = -1; H += H'
    c = zeros(9); c[8] = -1.
    Q = [H  c; c' 0.]
    push!(q_eqs, Q)
    H = zeros(9,9)
    H[1,5] = 1; H[2,4] = -1; H += H'
    c = zeros(9); c[9] = -1.
    Q = [H  c; c' 0.]
    push!(q_eqs, Q)

    # r2 x r3 - r1
    H = zeros(9,9)
    H[5,9] = 1; H[6,8] = -1; H += H'
    c = zeros(9); c[1] = -1.
    Q = [H  c; c' 0.]
    push!(q_eqs, Q)
    H = zeros(9,9)
    H[6,7] = 1; H[4,9] = -1; H += H'
    c = zeros(9); c[2] = -1.
    Q = [H  c; c' 0.]
    push!(q_eqs, Q)
    H = zeros(9,9)
    H[4,8] = 1; H[5,7] = -1; H += H'
    c = zeros(9); c[3] = -1.
    Q = [H  c; c' 0.]
    push!(q_eqs, Q)

    # r3 x r1 - r2
    H = zeros(9,9)
    H[8,3] = 1; H[9,2] = -1; H += H'
    c = zeros(9); c[4] = -1.
    Q = [H  c; c' 0.]
    push!(q_eqs, Q)
    H = zeros(9,9)
    H[9,1] = 1; H[7,3] = -1; H += H'
    c = zeros(9); c[5] = -1.
    Q = [H  c; c' 0.]
    push!(q_eqs, Q)
    H = zeros(9,9)
    H[7,2] = 1; H[8,1] = -1; H += H'
    c = zeros(9); c[6] = -1.
    Q = [H  c; c' 0.]
    push!(q_eqs, Q)

    return q_eqs
end


"""
    solvePACE_SDP(prob, y, weights[, λ=0.])

Solve shape & pose estimation problem `prob` given measurements `y`
with weight `weights`. Nonzero `λ` needed if more shapes than keypoints.

Uses first order convex relaxation, which finds certifially globally optimal solutions.

# Returns:
- `sol`: solution data
- `opt`: optimal cost
- `status`: solution status
- `gap`: suboptimality gap
"""
function solvePACE_SDP(prob, y, weights, λ=0.; silent=true)
    ## Setup
    K = prob.K

    if (sum(weights .!= 0) <= prob.K) && (λ == 0)
        λ = 0.1
        # @warn "overriding λ = 0.1 since N ≤ K."
    end

    ## eliminate position
    ȳ = sum(weights .* eachcol(y)) ./ sum(weights)
    B̄ = sum(weights .* eachslice(prob.B,dims=2)) ./ sum(weights)

    ## eliminate shape
    B̄s = (eachslice(prob.B,dims=2) .- [B̄]).*sqrt.(weights)
    ȳs = reduce(hcat, (eachcol(y) .- [ȳ]).*sqrt.(weights))
    B̂² = sum(transpose.(B̄s) .* B̄s)
    A = 2*(B̂² + λ*I)
    invA = inv(A)
    # Schur complement
    C1 = 2*(invA - invA*ones(K)*inv(ones(K)'*invA*ones(K))*ones(K)'*invA)
    c2 = invA*ones(K)*inv(ones(K)'*invA*ones(K))

    ## SDP
    model = Model(Mosek.Optimizer)
    @variable(model, X[1:10,1:10] ∈ PSDCone())

    # objective (F*vec(R) + g)'*(F*vec(R) + g)
    kronsum = sum(kron.(eachcol(ȳs), B̄s))'*PERMUTATION
    F = ((C1*kronsum)'*B̂² - 2*kronsum')*(C1*kronsum)
    g = (c2'*(-kronsum + B̂²*C1*kronsum))'
    if λ > 0
        g += (λ*c2'*C1*kronsum)'
        F += λ*(C1*kronsum)'*(C1*kronsum)
    end
    # objective matrix [vec(R); 1]'*C*[vec(R); 1]
    C = Symmetric([F  g;  g'  0])
    @objective(model, Min, tr(C*X))

    # SO(3) constraints
    for A in O3_CONSTRAINTS[1:end-1]
        @constraint(model, tr(A*X) == 0)
    end
    @constraint(model, X[end,end] == 1)
    # slightly less performant but doesn't make a huge difference
    for A in constraintsRHR()
        @constraint(model, tr(A*X) == 0)
    end

    ## solve
    if silent
        set_silent(model)
    end
    optimize!(model)
    opt = objective_value(model)

    # extract solution
    X_val = value.(X)
    
    # can do eigenvec decomp here but this is faster
    R_est = project2SO3(reshape(X_val[1:9,end], 3,3))
    c_est = reshape(C1*sum(transpose.(B̄s) .* [R_est'] .* eachcol(ȳs)) + c2,prob.K,1)
    p_est = reshape(ȳ,3,1) - R_est*B̄*c_est
    soln = Solution(c_est, p_est, R_est)

    # evaluate gap
    obj_est = [vec(R_est); 1]'*C*[vec(R_est); 1]
    gap = (obj_est - opt) / (opt+1)

    status = LOCAL_SOLUTION
    if gap < 1e-3
        status = GLOBAL_CERTIFIED
    end

    return soln, opt, status, gap
end


"""
    solvePACE_Manopt(prob, y, weights[, λ=0.])

Solve shape & pose estimation problem `prob` given measurements `y`
with weight `weights`. Nonzero `λ` needed if more shapes than keypoints.

Fast local solutions from initial guess `R0`, but no guarantee of global optima.

# Returns:
- `sol`: solution data
- `opt`: optimal cost
- `status`: solution status
"""
function solvePACE_Manopt(prob, y, weights, λ=0.; R0=nothing)
    ## Setup
    K = prob.K

    if (sum(weights .!= 0) <= prob.K) && (λ == 0)
        λ = 0.1
        # @warn "overriding λ = 0.1 since N ≤ K."
    end

    ## eliminate position
    ȳ = sum(weights .* eachcol(y)) ./ sum(weights)
    B̄ = sum(weights .* eachslice(prob.B,dims=2)) ./ sum(weights)

    ## eliminate shape
    B̄s = (eachslice(prob.B,dims=2) .- [B̄]).*sqrt.(weights)
    ȳs = reduce(hcat, (eachcol(y) .- [ȳ]).*sqrt.(weights))
    B̂² = sum(transpose.(B̄s) .* B̄s)
    A = 2*(B̂² + λ*I)
    invA = inv(A)
    # Schur complement
    C1 = 2*(invA - invA*ones(K)*inv(ones(K)'*invA*ones(K))*ones(K)'*invA)
    c2 = invA*ones(K)*inv(ones(K)'*invA*ones(K))

    ## JuMP model
    SO3 = Manifolds.Rotations(3)
    model = Model()
    @variable(model, R[1:3,1:3] in SO3)
    if isnothing(R0) R0 = randrotation() end
    set_start_value.(R, R0)
    set_optimizer(model, Manopt.JuMP_Optimizer)

    # objective (F*vec(R) + g)'*(F*vec(R) + g)
    kronsum = sum(kron.(eachcol(ȳs), B̄s))'*PERMUTATION
    F = ((C1*kronsum)'*B̂² - 2*kronsum')*(C1*kronsum)
    g = (c2'*(-kronsum + B̂²*C1*kronsum))'
    if λ > 0
        g += (λ*c2'*C1*kronsum)'
        F += λ*(C1*kronsum)'*(C1*kronsum)
    end
    # objective matrix [vec(R); 1]'*C*[vec(R); 1]
    C = Symmetric([F  g;  g'  0])
    @objective(model, Min, [vec(R); 1]'*C*[vec(R); 1])


    # Solve with JuMP
    optimize!(model)

    # Convert to solution
    R_est = value.(R)
    R_est = project2SO3(R_est)
    c_est = reshape(C1*sum(transpose.(B̄s) .* [R_est'] .* eachcol(ȳs)) + c2,prob.K,1)
    p_est = reshape(ȳ,3,1) - R_est*B̄*c_est
    soln = Solution(c_est, p_est, R_est)

    return soln, objective_value(model), LOCAL_SOLUTION
end


"""
    skew(y)

Converts vector `y ∈ R³` to skew symmetric matrix.
"""
function skew(y)
    [0. -y[3] y[2]; y[3] 0. -y[1]; -y[2] y[1] 0]
end

"""
    solvePACE_GN(prob, y, weights[, λ=0.; R₀, λ_lm = 0.])

Solve shape & pose estimation problem `prob` given measurements `y`
with weight `weights`. Nonzero `λ` needed if more shapes than keypoints.

Gauss-Newton specialized to PACE. Set `linesearch=true` to use line search for
best update and use `λ_lm` to do Levenberg-Marquardt. This is a local solver starting at `R₀`.
"""
function solvePACE_GN(prob, y, weights, λ=0.; R₀=nothing, λ_lm=0., max_iters=250)
    ## Setup
    K = prob.K

    if (sum(weights .!= 0) <= prob.K) && (λ == 0)
        λ = 0.1
        # @warn "overriding λ = 0.1 since N ≤ K."
    end

    ## eliminate position
    ȳ = sum(weights .* eachcol(y)) ./ sum(weights)
    B̄ = sum(weights .* eachslice(prob.B,dims=2)) ./ sum(weights)

    ## eliminate shape
    B̄s = (eachslice(prob.B,dims=2) .- [B̄]).*sqrt.(weights)
    ȳs = reduce(hcat, (eachcol(y) .- [ȳ]).*sqrt.(weights))
    B̂² = sum(transpose.(B̄s) .* B̄s)
    A = 2*(B̂² + λ*I)
    invA = inv(A)
    # Schur complement
    C1 = 2*(invA - invA*ones(K)*inv(ones(K)'*invA*ones(K))*ones(K)'*invA)
    c2 = invA*ones(K)*inv(ones(K)'*invA*ones(K))

    # optimal shape given rotation
    c(R) = C1*sum(transpose.(B̄s) .* [R'] .* eachcol(ȳs)) + c2
    # optimal position given shape & rotation
    p(R,c) = ȳ - R*B̄*c

    # registration residual
    # ri(R, i) = R'*ȳs[:,i] - B̄s[i]*c(R)
    # shape residual
    # rc(R) = √λ*(c(R))

    # Registration Jacobian
    # Ji(R, i) = R'*skew(ȳs[i]) - 2*B̄s[i]*G*sum([B̄s[j]'*R'*skew(ȳs[j]) for j = 1:prob.N])
    # Ji(R, i) = R'*skew(ȳs[:,i]) - 2*B̄s[i]*C1*sum(transpose.(B̄s) .* [R'] .* skew.(eachcol(ȳs)))

    # Shape jacobian
    # Jc(R) = 2*√λ*G*sum([B̄s[i]'*R'*skew(ȳs[i]) for i = 1:prob.N])
    # Jc(R) = 2*√λ*C1*sum(transpose.(B̄s) .* [R'] .* skew.(eachcol(ȳs)))

    ## G-N / L-M
    # initial condition
    if isnothing(R₀) R₀ = randrotation() end

    # iterate
    R_cur = R₀
    for i = 1:max_iters
        # registration residuals & Jacobians
        Jis = [R_cur'] .* skew.(eachcol(ȳs)) - 2*B̄s .* [C1*sum(transpose.(B̄s) .* [R_cur'] .* skew.(eachcol(ȳs)))]
        ris = [R_cur'] .* eachcol(ȳs) - B̄s .* [c(R_cur)]
        
        # shape residual & Jacobian
        if λ != 0
            rc = √λ*(c(R_cur))
            Jc = 2*√λ*C1*sum(transpose.(B̄s) .* [R_cur'] .* skew.(eachcol(ȳs)))
        else
            rc = zeros(K)
            Jc = zeros(K,3)
        end

        Σ = Jc'*Jc + sum(transpose.(Jis) .* Jis)
        v = Jc'*rc + sum(transpose.(Jis) .* ris)

        # TODO: this part could be better optimized (use dots, remove functions)
        # Σ = Jc(R_cur)'*Jc(R_cur) + sum([Ji(R_cur,i)'*Ji(R_cur,i) for i = 1:prob.N])
        # v = Jc(R_cur)'*rc(R_cur) + sum([Ji(R_cur,i)'*ri(R_cur,i) for i = 1:prob.N])
        δθ = -pinv(Σ + λ_lm*I)*v
        α  = 1.0
        R_cur = exp(skew(α*δθ))*R_cur

        # check for convergence
        if norm(δθ) < 1e-3
            # println("G-N Convergence.")
            break
        end
    end

    # Convert to solution
    R_est = project2SO3(R_cur)
    c_est = c(R_est)
    p_est = p(R_est, c_est)
    soln = Solution(reshape(c_est,K,1), reshape(p_est,3,1), R_est)

    # obj
    # obj = rc(R_cur)'*rc(R_cur) + sum([ri(R_cur, i)'*ri(R_cur,i) for i = 1:prob.N])
    rc = (λ == 0) ? zeros(K) : √λ*(c_est)
    ris = [R_cur'] .* eachcol(ȳs) - B̄s .* [c_est]
    obj = rc'*rc + sum(transpose.(ris) .* ris)

    return soln, obj
end

# TODO: MAKE FASTER