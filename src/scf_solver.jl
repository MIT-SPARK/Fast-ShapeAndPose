## SCF solver for category-level shape and pose estimation
# Lorenzo Shaikewitz, 6/26/2025

# SCF status
@enum StatusSCF begin
    FAILED
    LOCAL_SOLUTION
    GLOBAL_CERTIFIED
end


# permutation matrix vec(R') => vec(R)
const PERMUTATION = SMatrix{9,9}(hcat(vcat(1,zeros(8)),
                                       vcat(zeros(3), 1, zeros(5)),
                                       vcat(zeros(6), 1, zeros(2)),
                                       vcat(0,1,zeros(7)),
                                       vcat(zeros(4),1,zeros(4)),
                                       vcat(zeros(7),1,0),
                                       vcat(zeros(2),1,zeros(6)),
                                       vcat(zeros(5),1,zeros(3)),
                                       vcat(zeros(8),1)
                                       )...)

"""
    solvePACE_SCF(prob, y, weights[...])

Hyper-fast local solutions with initial guess

# Arguments:
- `prob`: shape & pose estimation problem data
- `y`: keypoint measurements [3 x N]
- `weights`: keypoint weights [N]
- `λ=0`: shape regularization (must be >0 if N ≤ K)
## Optional:
- `max_iters=250`: iterations to terminate after
- `q0=nothing`: intial guess (default is random)
- `certify=false`: whether to call certifier
- `tol=1e-6`: termination numerical tolerance

# Returns:
- `sol`: solution data
- `opt`: optimal cost
- `status`: solution status
- `scf_iters`: number of solver iterations
"""
function solvePACE_SCF(prob::Problem, y, weights, λ=0.;
                       max_iters::Int=150, q0=nothing, certify=false, tol=1e-6)

    ## Setup
    K = prob.K

    if (sum(weights .!= 0) <= prob.K) && (λ == 0)
        λ = 0.1
        # @warn "overriding λ = 0.1 since N ≤ K."
        # TODO: better thing to set lambda to?
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

    ## write Lagrangian
    # constant term
    # L = sum([ȳs[:,i]'*ȳs[:,i] for i = 1:N])
    L = sum(transpose.(eachcol(ȳs)) .* eachcol(ȳs))
    L += c2'*B̂²*c2
    L += λ*(c2'*c2)
    # quadratic in q
    if λ > 0
        # L123 = 4*sum([Ω1(ȳs[:,i])*Ω2(B̄s[i]*(I-λ*C1-C1*B̂²)*c2) for i = 1:N])
        L123 = Symmetric(SMatrix{4,4}(4*sum(Ω1.(eachcol(ȳs)) .* Ω2.(B̄s .* [(I-λ*C1-C1*B̂²)*c2]))...))
    else
        # L123 = 4*sum([Ω1(ȳs[:,i])*Ω2(B̄s[i]*c2) for i = 1:N])
        L123 = Symmetric(SMatrix{4,4}(4*sum(Ω1.(eachcol(ȳs)) .* Ω2.(B̄s.*[c2]))...))
    end

    ## solve via self-consistent field iteration
    # initial guess
    if isnothing(q0)
        # uniform random sample
        q0 = normalize(@SArray randn(4))
    end

    # SCF
    q_scf = q0
    scf_iters = max_iters
    opt = nothing
    for iter = 1:max_iters
        # compute lagrangian
        R_scf = quat2rotm(q_scf)
        mat = L123 + Symmetric(SMatrix{4,4}(4*sum(Ω1.(eachcol(ȳs)) .* Ω2.(B̄s .* [C1*(2*I-B̂²*C1-λ*C1)*sum(transpose.(B̄s).*[R_scf'].*eachcol(ȳs))]) )...))

        # mat = ℒ(q_scf)
        q_new = eigvecs(mat)[:,1] # accelerate?
        q_new = normalize(q_new)

        # termination: 
        if abs(abs(q_scf'*q_new) - 1) < tol
            opt = q_new'*(1/4*(L123) + 1/4*mat)*q_new + L
            scf_iters = iter
            q_scf = q_new
            break
        end
        q_scf = q_new
    end

    ## compute solution
    R_est = quat2rotm(q_scf)
    # c_est = reshape(C1*sum([B̄s[i]'*R_est'*ȳs[:,i] for i = 1:prob.N]) + c2,prob.K,1)
    c_est = reshape(C1*sum(transpose.(B̄s) .* [R_est'] .* eachcol(ȳs)) + c2,prob.K,1)
    p_est = reshape(ȳ,3,1) - R_est*B̄*c_est
    soln = Solution(c_est, p_est, R_est)

    ## check global optimality
    if certify
        # kronsum = sum([kron(ȳs[:,j]',B̄s[j]') for j = 1:N])*P
        kronsum = sum(kron.(eachcol(ȳs), B̄s))'*PERMUTATION

        # compute F and g of objective in form:
        # (F*vec(R) + g)'*(F*vec(R) + g)
        F = ((C1*kronsum)'*B̂² - 2*kronsum')*(C1*kronsum)
        g = (c2'*(-kronsum + B̂²*C1*kronsum))'
        if λ > 0
            g += (λ*c2'*C1*kronsum)'
            F += λ*(C1*kronsum)'*(C1*kronsum)
        end

        # objective matrix [vec(R); 1]'*C*[vec(R); 1]
        C = Symmetric([F  g;  g'  0])

        # lienar system for lagrange multipliers
        xlocal = [vec(soln.R); 1]
        AL = reduce(hcat, O3_CONSTRAINTS .* [xlocal])
        μ = AL \ (C*xlocal)
        # certificate matrix
        S = C - sum(μ .* O3_CONSTRAINTS)
    end


    ## report status
    if isnothing(opt)
        status = FAILED
        r = eachcol(y) - [soln.R].*(eachslice(prob.B, dims=2).*[soln.c]) .- [soln.p]
        opt = sum(weights .* (transpose.(r).*r))[1] + λ*(c_est[:,1]'*c_est[:,1])
    elseif certify && (eigvals(S)[1] > -1e-3)
        status = GLOBAL_CERTIFIED
    else
        status = LOCAL_SOLUTION
    end

    return soln, opt, status, scf_iters
end



"""
    solvePACE_SCF2(prob, y, weights[...])

Solve shape & pose estimation problem `prob` given measurements `y`
with weight `weights`. Nonzero `lam` needed if more shapes than keypoints.

Hyper-fast local solutions with initial guess, can rerun for global solution.

# Optional arguments:
- `local_iters`: maximum number of SCF iterations to run given initial guess.
    Failure is declared if a fixed point is not reached before this.
- `obj_thresh`: objective change threshold to terminate at. Not currently recommended.
- `q0`: initial quaternion guess. Must be normalized.
- `global_iters`: number of times to run for global solution search. Uses farthest point sampling.
- `debug`: return logs of all distinct minima and objective function.
- `all_logs`: return full logs of all runs.
"""
function solvePACE_SCF2(prob::Problem, y, weights, lam=0.; 
        local_iters=100, obj_thresh=0., q0=nothing,
        global_iters=15,
        debug=false, all_logs=false)

    # TODO: CLEAN UP IMPLEMENTATION
    # StaticArrays for faster
    # No bounds checking
    # More dots where possible (can I replace the sums?)
    # Use views for slices

    ## SETUP
    K = prob.K
    N = prob.N

    if sum(weights .!= 0) <= prob.K && lam == 0
        lam = 0.1
        @warn "setting lam = 0.1 since N ≤ K."
    end

    ## eliminate position
    ybar = sum(weights .* eachcol(y)) ./ sum(weights)
    Bbar = sum(weights .* eachslice(prob.B,dims=2)) ./ sum(weights)

    ## eliminate shape
    Bc = (eachslice(prob.B,dims=2) .- [Bbar]).*sqrt.(weights)
    yc = reduce(hcat, (eachcol(y) .- [ybar]).*sqrt.(weights))
    Bc2 = sum([Bc[i]'*Bc[i] for i = 1:N])
    A = 2*(Bc2 + lam*I)
    invA = inv(A)
    # Schur complement
    c1 = 2*(invA - invA*ones(K)*inv(ones(K)'*invA*ones(K))*ones(K)'*invA)
    c2 = invA*ones(K)*inv(ones(K)'*invA*ones(K))

    ## Lagrangian terms
    # constant
    L = sum([yc[:,i]'*yc[:,i] for i = 1:N])
    L += c2'*Bc2*c2
    L += lam*(c2'*c2)
    # quadratic in q
    if lam > 0
        L123 = 4*sum([Ω1(yc[:,i])*Ω2(Bc[i]*(I-lam*c1-c1*Bc2)*c2) for i = 1:N])
    else
        L123 = 4*sum([Ω1(yc[:,i])*Ω2(Bc[i]*c2) for i = 1:N])
    end
    # quartic in q
    function Lquartic(q)
        R = quat2rotm(q)
        Bry = sum([Bc[j]'*R'*yc[:,j] for j = 1:N])
        return 4*sum([Ω1(yc[:,i])*Ω2(Bc[i]*c1*(2*I-Bc2*c1-lam*c1)*Bry) for i = 1:N])
    end

    # objective
    obj(q) = q'*(1/2*(L123) + 1/4*Lquartic(q))*q + L
    obj(q, mat) = q'*(1/4*(L123) + 1/4*mat)*q + L
    # Lagrangian
    ℒ(q) = Symmetric(L123 + Lquartic(q))
    
    ## SOLVE
    # self-consistent field iteration function
    function scf(q_scf; quat_log=nothing)
        new = false
        log = [q_scf]
        obj_last = 1e6
        mat_last = zeros(2,4) # to throw errors if improper
        for iter = 1:Int(local_iters)
            if obj_thresh !=0 && iter > 5
                mat = mat_last
            else
                mat = ℒ(q_scf)
            end
            q_new = eigvecs(mat)[:,1]
            normalize!(q_new)
            push!(log, q_new)
    
            if !isnothing(quat_log) && !all_logs
                doublebreak = false
                for quat in quat_log
                    if abs(abs(q_new'*quat) - 1) < 1e-5
                        doublebreak = true
                        break
                    end
                end
                if doublebreak
                    break
                end
            end
    
            if abs(abs(q_scf'*q_new) - 1) < 1e-6
                q_scf = q_new
                new = true
                break
            end

            if obj_thresh != 0. && iter >= 5 # start early termination a little later
                mat_last = ℒ(q_new)
                obj_new = obj(q_new, mat_last)
                if abs(obj_new - obj_last) < obj_thresh
                    q_scf = q_new
                    new = true
                    break
                end
                obj_last = obj_new
            end
            q_scf = q_new
        end
        return q_scf, new, log
    end

    ## solve to optimality
    # sample on grid
    grid_pts = max(100, global_iters)
    q_guess = normalize.(eachcol(randn(4, grid_pts)))
    dists = 100*ones(grid_pts)
    # add initial guess
    if !isnothing(q0)
        prepend!(q_guess, [normalize(q0)])
        prepend!(dists, 101)
    end

    # Repeat and do farthest point sampling
    # TODO: can accelerate this
    q_scfs = []
    q_logs = []
    for idx = 1:global_iters
        q0_new = q_guess[argmax(dists)]
        # angular distance farther point sampling
        sines(qs, q) = [1. .- qs[i]'*q for i = 1:grid_pts] # TODO: this can be more efficient
        dists = min.(dists, sines(q_guess, q0_new))
        # dists = min.(dists, norm.(q_guess .- [q0_new]))
        q_scf, new, q_log = scf(q0_new; quat_log=q_scfs)
        if new
            push!(q_scfs, q_scf)
            push!(q_logs, q_log)
        end
    end
    if length(q_scfs) == 0
        printstyled("SCF Failed.\n", color=:red)
        R_est = quat2rotm(normalize(randn(4)))
        c_est = reshape(c1*sum([Bc[i]'*R_est'*yc[:,i] for i = 1:prob.N]) + c2,prob.K,1)
        p_est = ybar - R_est*Bbar*c_est
        soln = Solution(c_est, p_est, R_est)
        obj_val = obj(rotm2quat(R_est))
        if debug || all_logs
            return soln, obj_val, q_logs, ℒ, obj
        end
        if global_iters == 1
            return soln, obj_val, local_iters
        end
        return soln, obj_val
    end

    objs = obj.(q_scfs)
    minidx = argmin(objs)
    q_scf = q_scfs[minidx]
    obj_val = objs[minidx]

    # convert to solution
    R_est = quat2rotm(q_scf)
    c_est = reshape(c1*sum([Bc[i]'*R_est'*yc[:,i] for i = 1:prob.N]) + c2,prob.K,1)
    p_est = ybar - R_est*Bbar*c_est

    soln = Solution(c_est, p_est, R_est)

    if debug || all_logs
        return soln, obj_val, q_logs, ℒ, obj
    end
    if global_iters == 1
        return soln, obj_val, length(q_logs[1])
    end
    return soln, obj_val
end