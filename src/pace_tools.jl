## Tools for generating and solving shape and pose estimation problems
# Lorenzo Shaikewitz

### SOLVERS
# TSSOS: SDP relaxation
# Manopt: local on-manifold
# SCF: quaternion-based self-consistent field iteration

using Random
using Statistics
using LinearAlgebra
using Printf

using LorenzoRotations

struct Problem
    N ::Int64       # num keypoints
    K ::Int64       # num shapes
    σm::Float64     # measurement noise standard deviation
    r ::Float64     # shape radius
    B ::Array{Float64, 3} # shape library
end

struct Solution
    c ::Matrix{Float64} # gt shape coefficent
    p ::Matrix{Float64} # gt position
    R ::Matrix{Float64} # gt rotation
end

"""
    genproblem(;N::Int64=10, K::Int64=4, σm::Float64=0.05, r::Float64=0.2)

Generate a shape & pose estimation problem with `N` keypoints & `K` models.

Set Gaussian keypoint noise with `σm` and shape radius with `r`.
"""
function genproblem(;N::Int64=10, K::Int64=4, σm::Float64=0.05, r::Float64=0.2)
    # generate shape lib
    meanShape = randn(3, N)
    meanShape .-= mean(meanShape,dims=2)
    
    shapes = zeros(3,N,K)
    for k = 1:K
        shapes[:,:,k] = meanShape + r * randn(3,N)
    end
    B = reshape(shapes, 3*N, K)
    prob = Problem(N, K, σm, r, shapes)

    # gt shape
    c = rand(K,1)
    c /= sum(c)

    # gt position
    p = randn(3,1) .+ 1.0

    # gt rotation
    R = randrotation()
    # save
    gt = Solution(c, p, R)

    # convert to measurements
    shape = reshape(B*c, (3,N))
    y = R*shape .+ p .+ σm*randn(3,N)

    # save
    return prob, gt, y
end

### TSSOS SOLVER
using TSSOS
using DynamicPolynomials

"""
    solvePACE_TSSOS(prob, y, weights[, lam=0.])

Solve shape & pose estimation problem `prob` given measurements `y`
with weight `weights`. Nonzero `lam` needed if more shapes than keypoints.

Uses first order convex relaxation, which finds certifially globally optimal solutions.
"""
function solvePACE_TSSOS(prob, y, weights, lam=0.)
    @polyvar R[1:3,1:3]
    vars = vec(R)

    if sum(weights .!= 0) <= prob.K
        lam = 0.01
    end

    # symbolic expressions for c, p
    B = reshape(prob.B, 3*prob.N, prob.K)
    W = Diagonal(repeat(weights, inner=3))
    yw = zeros(3,1)
    Bw = zeros(3,prob.K)
    for i = 1:prob.N
        yw += weights[i]*y[:,i]
        Bw += weights[i]*B[(1+3*(i-1)):(3*i), :]
    end
    yw /= sum(weights)
    Bw /= sum(weights)
    Bbar = B - repeat(Bw, outer=prob.N)
    H11 = 2*(Bbar'*W*Bbar + lam*I)
    invH11 = inv(H11)
    H12 = ones(prob.K,1)
    invS = inv(-H12'*invH11*H12)
    G = invH11 + invH11*H12*invS*H12'*invH11
    g = -invH11*H12*invS

    sumh = zeros(Polynomial{true, Float64}, 3*prob.N, 1)
    for i = 1:prob.N
        sumh[1+3*(i-1):3*i] = weights[i]*R'*(y[:,i] - yw)
    end

    cbar = 1/prob.K * ones(prob.K,1)

    c = G*(2*Bbar'*sumh + 2*lam*cbar) + g;

    p = yw - R*Bw*c

    obj = lam*(c - cbar)'*(c - cbar)
    for i = 1:prob.N
        term = R'*(y[:,i] - yw) - (B[1+3*(i-1):3*i,:] - Bw)*c
        obj += weights[i] * (term' * term)
    end
    # remove "matrix" polynomial
    obj = obj[1]

    eq = zeros(Polynomial{true, Float64}, 0)
    # R ∈ SO(3)
    append!(eq, vec(R'*R - I)) # O(3)
    append!(eq, R[1:3,3] .- cross(R[1:3,1],R[1:3,2]))
    append!(eq, R[1:3,1] .- cross(R[1:3,2],R[1:3,3]))
    append!(eq, R[1:3,2] .- cross(R[1:3,3],R[1:3,1]))

    # Solve with TSSOS
    pop = [obj; eq]
    opt, sol, data = tssos_first(pop, vars, 1, numeq=length(eq), solve=true, quotient=false, QUIET=true)

    # Convert to solution
    # Extract solution (largest eigenvec)
    mom = data.moment[1]
    r = sqrt(eigvals(mom)[end])*eigvecs(mom)[:,end]
    r /= r[1]
    R_est = reshape(r[2:end], (3,3))
    R_est = project2SO3(R_est)
    s(v) = subs(v,vars=>vec(R_est))
    c_est = s.(c)
    p_est = s.(p)

    soln = Solution(c_est, p_est, R_est)
    
    # evaluate gap
    obj_est = obj(vec(R) => vec(soln.R))
    gap = (obj_est - opt) / (opt+1)

    return soln, opt, gap
end

### Manopt (LOCAL) SOLVER
using JuMP
import Manifolds
import Manopt
"""
    solvePACE_Manopt(prob, y, weights[, lam=0.])

Solve shape & pose estimation problem `prob` given measurements `y`
with weight `weights`. Nonzero `lam` needed if more shapes than keypoints.

Fast local solutions from initial guess `R0`, but no guarantee of global optima.
"""
function solvePACE_Manopt(prob, y, weights, lam=0.; R0=nothing)
    SO3 = Manifolds.Rotations(3)

    if sum(weights .!= 0) <= prob.K
        lam = 0.01
    end

    model = Model()
    @variable(model, R[1:3,1:3] in SO3)
    if isnothing(R0) R0 = randrotation() end
    set_start_value.(R, R0)
    set_optimizer(model, Manopt.JuMP_Optimizer)

    # symbolic expressions for c, p
    B = reshape(prob.B, 3*prob.N, prob.K)
    W = Diagonal(repeat(weights, inner=3))
    yw = zeros(3,1)
    Bw = zeros(3,prob.K)
    for i = 1:prob.N
        yw += weights[i]*y[:,i]
        Bw += weights[i]*B[(1+3*(i-1)):(3*i), :]
    end
    yw /= sum(weights)
    Bw /= sum(weights)
    Bbar = B - repeat(Bw, outer=prob.N)
    H11 = 2*(Bbar'*W*Bbar + lam*I)
    invH11 = inv(H11)
    H12 = ones(prob.K,1)
    invS = inv(-H12'*invH11*H12)
    G = invH11 + invH11*H12*invS*H12'*invH11
    g = -invH11*H12*invS

    sumh = zeros(AffExpr, 3*prob.N, 1)
    for i = 1:prob.N
        sumh[1+3*(i-1):3*i] = weights[i]*R'*(y[:,i] - yw)
    end

    cbar = 1/prob.K * ones(prob.K,1)

    c = G*(2*Bbar'*sumh + 2*lam*cbar) + g;

    p = yw - R*Bw*c

    obj = lam*(c - cbar)'*(c - cbar)
    for i = 1:prob.N
        term = R'*(y[:,i] - yw) - (B[1+3*(i-1):3*i,:] - Bw)*c
        obj += weights[i] * (term' * term)
    end
    # remove "matrix" polynomial
    obj = obj[1]
    @objective(model, Min, obj)

    # equality constraints
    # R ∈ SO(3)
    # @constraint(model, vec(R'*R - I) .== zeros(9))
    # @constraint(model, R[1:3,3] .== cross(R[1:3,1],R[1:3,2]))
    # @constraint(model, R[1:3,1] .== cross(R[1:3,2],R[1:3,3]))
    # @constraint(model, R[1:3,2] .== cross(R[1:3,3],R[1:3,1]))

    # Solve with JuMP
    optimize!(model)

    # Convert to solution
    R_est = value.(R)
    R_est = project2SO3(R_est)
    c_est = value.(c)
    p_est = value.(p)

    soln = Solution(c_est, p_est, R_est)

    return soln, value.(obj)
end


### QUATERNION SOLVER (just SCF)
"""
    Ω1(q)

Defined by `a ⊗ b = Ω1(a)*b` where `a,b` are quaternions.

Automatically adds leading 0 if dimension is 3.

Also satisfies `Ω1(a^{-1}) = Ω1(a)^T`.
"""
function Ω1(q)
    if size(q)[1] == 3
        q = [0; q]
    end
    [q[1] -q[2] -q[3] -q[4];
     q[2]  q[1] -q[4]  q[3];
     q[3]  q[4]  q[1] -q[2];
     q[4] -q[3]  q[2]  q[1]]
end

"""
    Ω2(q)

Defined by `a ⊗ b = Ω2(b)*a` where `a,b` are quaternions.

Automatically adds leading 0 if dimension is 3.

Also satisfies `Ω2(a^{-1}) = Ω2(a)^T`.
"""
function Ω2(q)
    if size(q)[1] == 3
        q = [0; q]
    end
    [q[1] -q[2] -q[3] -q[4];
     q[2]  q[1]  q[4] -q[3];
     q[3] -q[4]  q[1]  q[2];
     q[4]  q[3] -q[2]  q[1]]
end


"""
    solvePACE_SCF(prob, y, weights[...])

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
function solvePACE_SCF(prob::Problem, y, weights, lam=0.; 
        local_iters=100, obj_thresh=0., q0=nothing,
        global_iters=15,
        debug=false, all_logs=false)
    ## SETUP
    K = prob.K
    N = prob.N

    if sum(weights .!= 0) <= prob.K
        lam = 0.1
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
        if obj_thresh != 0
            mat_last = ℒ(q_scf)
        end
        for _ = 1:Int(local_iters)
            if obj_thresh !=0
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

            if obj_thresh != 0.
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


"""
    solvePACE_Power(prob, y, weights[, lam=0.; grid=100, local_iters=100, global_iters=15])

Solve shape & pose estimation problem `prob` given measurements `y`
with weight `weights`. Nonzero `lam` needed if more shapes than keypoints.

Hyper-fast local solutions with initial guess. `local_iters` controls 
max iterations to look for local solution. Uses power method instead of SCF.

For global solutions, `grid` discretizes the space and `global_iters` 
dictates how many initial conditions to try.
"""
function solvePACE_Power(prob, y, weights, lam=0.; q0=nothing, grid=100, local_iters=400, global_iters=15, logs=false, all_logs=false, objthresh=0.)
    ## SETUP
    K = prob.K
    N = prob.N

    if sum(weights .!= 0) <= prob.K
        lam = 0.1
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
    # Lagrangian
    ℒ(q) = L123 + Lquartic(q)
    
    ## SOLVE
    # self-consistent field iteration function
    function power(q_power; quat_log=nothing)
        new = false
        log = [q_power]
        for _ = 1:Int(local_iters)
            mat = ℒ(q_power)
            q_new = normalize(mat*q_power)
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
    
            if abs(abs(q_power'*q_new) - 1) < 1e-6
                q_power = q_new
                new = true
                break
            end

            if objthresh != 0.
                if abs(obj(q_new) - obj(q_power)) < objthresh
                    q_power = q_new
                    new = true
                    break
                end
            end

            q_power = q_new
        end
        return q_power, new, log
    end

    # solve to optimality
    q_guess = normalize.(eachcol(randn(4, grid)))
    dists = 100*ones(grid)

    # Repeat and do farthest point sampling
    # TODO: can accelerate this
    q_powers = []
    q_logs = []
    for idx = 1:min(grid, global_iters)
        q0_new = q_guess[argmax(dists)]
        dists = min.(dists, norm.(q_guess .- [q0_new]))
        
        q_power, new, q_log = power(q0_new; quat_log=q_powers)
        if new
            push!(q_powers, q_power)
            push!(q_logs, q_log)
        end
    end
    if length(q_powers) == 0
        printstyled("Power Method Failed.\n", color=:red)
        soln = Solution(zeros(prob.K,1), zeros(3,1), zeros(3,3))
        obj_val = 1e6
        if logs
            return soln, obj_val, q_logs, ℒ, obj
        end
        if global_iters == 1
            return soln, obj_val, local_iters
        end
        return soln, obj_val
    end

    objs = obj.(q_powers)
    minidx = argmin(objs)
    q_power = q_powers[minidx]
    obj_val = objs[minidx]

    # convert to solution
    R_est = quat2rotm(q_power)
    c_est = reshape(c1*sum([Bc[i]'*R_est'*yc[:,i] for i = 1:prob.N]) + c2,prob.K,1)
    p_est = ybar - R_est*Bbar*c_est

    soln = Solution(c_est, p_est, R_est)

    if logs
        return soln, obj_val, q_logs, ℒ, obj
    end
    if global_iters == 1
        return soln, obj_val, length(q_logs[1])
    end
    return soln, obj_val
end

"""
Old version of solvePACE_SCF, kept currently for debugging.
"""
function solvePACE_SCF_OLD(prob, y, weights, lam=0.; grid=100, local_iters=100, global_iters=15, logs=false, all_logs=false, objthresh=0.)
    ## SETUP
    K = prob.K
    N = prob.N

    if sum(weights .!= 0) <= prob.K
        lam = 0.1
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
    L1 = zeros(4,4)
    L2 = zeros(4,4)
    L3 = zeros(4,4)
    for i = 1:N
        L1 += -2*Ω1(yc[:,i])'*Ω2(Bc[i]*c2)
        if lam > 0.
            L3 +=  2*lam*Ω1(yc[:,i])'*Ω2(Bc[i]*c1'*c2)
            L2 +=  2*Ω1(yc[:,i])'*Ω2(Bc[i]*c1'*Bc2*c2) # (c1'*Bc2*c2 = 0 if lam=0)
        end
    end
    L1 += L1'
    L2 += L2'
    L3 += L3'
    # quartic in q
    function Lquartic(q)
        L4 = zeros(4,4)
        L5 = zeros(4,4)
        L6 = zeros(4,4)
        R = quat2rotm(q)
        Bry = sum([Bc[j]'*R'*yc[:,j] for j = 1:N])
        for i = 1:N
            if lam > 0.
                L4 += 2*lam*Ω1(yc[:,i])'*Ω2(Bc[i]*c1'*c1*Bry)
            end
            L5 += -2*Ω1(yc[:,i])'*Ω2(Bc[i]*(c1+c1')*Bry)
            L6 += 2*Ω1(yc[:,i])'*Ω2(Bc[i]*c1'*Bc2'*c1*Bry)
        end
        L4 += L4'
        L5 += L5'
        L6 += L6'
        return L4 + L5 + L6
    end

    # objective
    obj(q) = q'*(1/2*(L1 + L2 + L3) + 1/4*Lquartic(q))*q + L
    # Lagrangian
    ℒ(q) = Symmetric(L1 +L2 + L3 + Lquartic(q))
    
    ## SOLVE
    # self-consistent field iteration function
    function scf(q_scf; quat_log=nothing)
        new = false
        log = [q_scf]
        for _ = 1:Int(local_iters)
            mat = ℒ(q_scf)
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

            if objthresh != 0.
                if abs(obj(q_new) - obj(q_scf)) < objthresh
                    q_scf = q_new
                    new = true
                    break
                end
            end
            q_scf = q_new
        end
        return q_scf, new, log
    end

    # solve to optimality
    q_guess = normalize.(eachcol(randn(4, grid)))
    dists = 100*ones(grid)

    # Repeat and do farthest point sampling
    # TODO: can accelerate this
    q_scfs = []
    q_logs = []
    for idx = 1:min(grid, global_iters)
        q0_new = q_guess[argmax(dists)]
        dists = min.(dists, norm.(q_guess .- [q0_new]))
        
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
        if logs
            return soln, obj_val, q_logs, ℒ, obj
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

    if logs
        return soln, obj_val, q_logs, ℒ, obj
    end
    return soln, obj_val
end


### Gauss-Newton solver
function skew(y)
    [0. -y[3] y[2]; y[3] 0. -y[1]; -y[2] y[1] 0]
end

"""
    solvePACE_GN(prob, y, weights[, lam=0.; R₀, λ = 0.])

Solve shape & pose estimation problem `prob` given measurements `y`
with weight `weights`. Nonzero `lam` needed if more shapes than keypoints.

Gauss-Newton specialized to PACE. Set `linesearch=true` to use line search for
best update and use `λ` to do Levenberg-Marquardt. This is a local solver starting at `R₀`.
"""
function solvePACE_GN(prob, y, weights, lam=0.; R₀=nothing, λ=0.)
    # initial condition
    if isnothing(R₀) R₀ = randrotation() end

    if sum(weights .!= 0) <= prob.K
        lam = 0.01
    end

    # symbolic expressions for c, p
    ybar = sum(weights .* eachcol(y)) ./ sum(weights)
    ytild = sqrt.(weights) .* eachcol(y .- ybar)
    Bbar = sum(weights .* eachslice(prob.B,dims=2)) ./ sum(weights)
    Btild = sqrt.(weights) .* (eachslice(prob.B,dims=2) .- [Bbar])
    H11 = 2*lam*I + 2*sum([Btild[i]'*Btild[i] for i = 1:prob.N])
    invH11 = inv(H11)
    H12 = ones(prob.K,1)
    invS = inv(-H12'*invH11*H12)
    G = invH11 + invH11*H12*invS*H12'*invH11
    g = -invH11*H12*invS

    # optimal shape given rotation
    cbar = 1/prob.K * ones(prob.K,1)
    function c(R)
        G*(2*lam*cbar + 2*sum([Btild[i]'*R'*ytild[i] for i = 1:prob.N])) + g
    end
    # optimal position given shape & rotation
    p(R,c) = ybar - R*reshape(Bbar,3,prob.K)*c

    # registration residual
    ri(R, i) = R'*ytild[i] - Btild[i]*c(R)
    # shape residual
    rc(R) = √lam*(c(R) - cbar)

    # Registration Jacobian
    Ji(R, i) = R'*skew(ytild[i]) - 2*Btild[i]*G*sum([Btild[j]'*R'*skew(ytild[j]) for j = 1:prob.N])

    # Shape jacobian
    Jc(R) = 2*√lam*G*sum([Btild[i]'*R'*skew(ytild[i]) for i = 1:prob.N])

    # G-N / L-M
    R_cur = R₀
    for i = 1:100
        Σ = Jc(R_cur)'*Jc(R_cur) + sum([Ji(R_cur,i)'*Ji(R_cur,i) for i = 1:prob.N])
        v = Jc(R_cur)'*rc(R_cur) + sum([Ji(R_cur,i)'*ri(R_cur,i) for i = 1:prob.N])
        δθ = -inv(Σ + λ*I)*v
        α  = 1.0
        R_cur = exp(skew(α*δθ))*R_cur

        # check for convergence
        if norm(δθ) < 1e-3
            # println("G-N Convergence.")
            break
        end
    end

    # obj
    obj = rc(R_cur)'*rc(R_cur) + sum([ri(R_cur, i)'*ri(R_cur,i) for i = 1:prob.N])

    # Convert to solution
    R_est = R_cur
    R_est = project2SO3(R_est)
    c_est = c(R_est)
    p_est = p(R_est, c_est)

    soln = Solution(c_est, p_est, R_est)

    return soln, value.(obj)
end