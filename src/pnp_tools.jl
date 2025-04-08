## Tools for generating and solving perspective-n-point problems
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

struct PnPProblem
    N ::Int64           # num points
    σm::Float64         # measurement noise standard deviation
    r ::Float64         # shape radius
    y ::Matrix{Float64} # 3D points
end
struct PnPSolution
    p ::Vector{Float64} # gt position
    R ::Matrix{Float64} # gt rotation
end

"""
    genpnpproblem(;N::Int64=10, σm::Float64=0.05, r::Float64=0.2)

Generate a PnP problem with `N` points, camera calibration matrix `K`, and keypoint measurement noise `σm`.

Camera calibration matrix is same as NOCS. Shape radius is `r`.
"""
function genPnPproblem(;N::Int64=10, σm::Float64=0.05, r::Float64=0.8)
    # gt 3D points
    y = r*randn(3, N)
    y .-= mean(y,dims=2)
    prob = PnPProblem(N, σm, r, y)

    # gt position
    p = randn(3,1) + [0; 0; 1.5]

    # gt rotation: make sure pointing towards object
    found_R = false
    pn = normalize(p)[:,1]
    R = I
    for _ = 1:100
        R = randrotation()
        if R[:,3]'*pn > cos(60*π/180.) # within 60 deg.
            found_R = true
            break
        end
    end
    if !found_R
        printstyled("Problem generation failed to find R! Suggest rerunning."; color=:red)
    end

    # save
    gt = PnPSolution(p[:,1], R)

    # convert to pixel measurements
    ϵ = σm*randn(3, N)
    u = R*y .+ p + ϵ
    u = reduce(hcat, eachcol(u) ./ u[3,:])

    # save
    return prob, gt, u
end

### TSSOS SOLVER
using TSSOS
using DynamicPolynomials

"""
    solvePnP_TSSOS(prob, u, weights)

Solve PnP estimation problem `prob` given pixel measurements `u`
with weight `weights`.

Uses first order convex relaxation, which finds certifially globally optimal solutions.
"""
function solvePnP_TSSOS(prob, u, weights)
    @polyvar R[1:3,1:3]
    vars = vec(R)

    # symbolic expression for p
    H = [sqrt(weights[i])*(u[:,i]*[0 0 1] - I) for i = 1:prob.N]
    sumHinv = inv(sum([(H[i]'*H[i]) for i = 1:prob.N]))
    p = -sumHinv*sum([H[i]'*H[i]*R*prob.y[:,i] for i = 1:prob.N])

    # objective
    obj = sum([(H[i]*(R*prob.y[:,i] + p))'*(H[i]*(R*prob.y[:,i] + p)) for i = 1:prob.N])

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
    p_est = s.(p)

    soln = PnPSolution(p_est, R_est)
    
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
    solvePnP_Manopt(prob, u, weights[, R0=rand])

Solve PnP problem `prob` given pixel measurements `u`
with weight `weights`.

Fast local solutions from initial guess `R0`, but no guarantee of global optima.
"""
function solvePnP_Manopt(prob, u, weights; R0=nothing)
    SO3 = Manifolds.Rotations(3)

    model = Model()
    @variable(model, R[1:3,1:3] in SO3)
    if isnothing(R0) R0 = randrotation() end
    set_start_value.(R, R0)
    set_optimizer(model, Manopt.JuMP_Optimizer)

    # symbolic expression for p
    H = [sqrt(weights[i])*(u[:,i]*[0 0 1] - I) for i = 1:prob.N]
    sumHinv = inv(sum([(H[i]'*H[i]) for i = 1:prob.N]))
    p = -sumHinv*sum([H[i]'*H[i]*R*prob.y[:,i] for i = 1:prob.N])

    # objective
    obj = sum([(H[i]*(R*prob.y[:,i] + p))'*(H[i]*(R*prob.y[:,i] + p)) for i = 1:prob.N])
    @objective(model, Min, obj)

    # Solve with JuMP
    optimize!(model)

    # Convert to solution
    R_est = value.(R)
    R_est = project2SO3(R_est)
    p_est = value.(p)

    soln = PnPSolution(p_est, R_est)

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
    solvePnP_SCF(prob, y, weights[,...])

Solve PnP problem `prob` given measurements `y`
with weight `weights`.

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
function solvePnP_SCF(prob::PnPProblem, u, weights; 
        local_iters=100, obj_thresh=0., q0=nothing,
        global_iters=15,
        debug=false, all_logs=false)
    
    ## eliminate position
    H = [weights[i]*Symmetric((u[:,i]*[0 0 1] - I)'*(u[:,i]*[0 0 1] - I)) for i = 1:prob.N]
    sumHinv = inv(sum(H))
    # p = -sumHinv*sum([H[i]*R*prob.y[:,i] for i = 1:prob.N])

    ## Lagrangian terms
    # constant
    # quadratic in q
    # quartic in q
    function Lquartic(q)
        R = quat2rotm(q)
        L1 = sum([-Ω1(H[i]*R*prob.y[:,i])*Ω2(prob.y[:,i]) for i = 1:prob.N])
        L2 = sum([sum([-Ω1(H[k]*sumHinv*H[i]*sumHinv*sum([H[j]*R*prob.y[:,j] for j = 1:prob.N]))*Ω2(prob.y[:,k]) for k = 1:prob.N]) for i = 1:prob.N])
        L3 = sum([sum([Ω1(2*H[k]*sumHinv*H[i]*R*prob.y[:,i])*Ω2(prob.y[:,k]) for k = 1:prob.N]) for i = 1:prob.N])
        return 4*(L1 + L2 + L3)
    end

    # objective
    obj(q) = 1/4*q'*Lquartic(q)*q
    # Lagrangian
    ℒ(q) = Symmetric(Lquartic(q) - I)
    
    ## SOLVE
    # self-consistent field iteration function
    function scf(q_scf; quat_log=nothing)
        new = false
        log = [q_scf]
        obj_last = 1e6
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

            if obj_thresh != 0.
                obj_new = obj(q_new)
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
        p_est = -sumHinv*sum([H[i]*R_est*prob.y[:,i] for i = 1:prob.N])
        soln = PnPSolution(p_est, R_est)
        obj_val = obj(rotm2quat(R_est))
        if debug || all_logs
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
    p_est = -sumHinv*sum([H[i]*R*prob.y[:,i] for i = 1:prob.N])

    soln = PnPSolution(p_est, R_est)

    if debug || all_logs
        return soln, obj_val, q_logs, ℒ, obj
    end
    return soln, obj_val
end

# prob, gt, u = genPnPproblem(σm = 0.1)
# weights = ones(prob.N)

# H = [weights[i]*Symmetric((u[:,i]*[0 0 1] - I)'*(u[:,i]*[0 0 1] - I)) for i = 1:prob.N]
# sumHinv = inv(sum(H))

# q = rotm2quat(gt.R)

# L1 = q'*sum([-Ω1(H[i]*gt.R*prob.y[:,i])*Ω2(prob.y[:,i]) for i = 1:prob.N])*q
# L1_check = sum([prob.y[:,i]'*gt.R'*H[i]*gt.R*prob.y[:,i] for i = 1:prob.N])

# L2 = q'*sum([sum([-Ω1(H[k]*sumHinv*H[i]*sumHinv*sum([H[j]*gt.R*prob.y[:,j] for j = 1:prob.N]))*Ω2(prob.y[:,k]) for k = 1:prob.N]) for i = 1:prob.N])*q
# L2_check = sum([sum([prob.y[:,k]'*gt.R'*H[k] for k = 1:prob.N])*sumHinv*H[i]*sumHinv*sum([H[j]*gt.R*prob.y[:,j] for j = 1:prob.N]) for i = 1:prob.N])

# L3 = q'*sum([sum([Ω1(2*H[k]*sumHinv*H[i]*gt.R*prob.y[:,i])*Ω2(prob.y[:,k]) for k = 1:prob.N]) for i = 1:prob.N])*q
# L3_2 = q'*sum([sum([Ω1(2*H[i]*sumHinv*H[k]*gt.R*prob.y[:,k])*Ω2(prob.y[:,i]) for k = 1:prob.N]) for i = 1:prob.N])*q
# L3_check = sum([sum([-2*prob.y[:,i]'*gt.R'*H[i]*sumHinv*H[k]*gt.R*prob.y[:,k] for k = 1:prob.N]) for i = 1:prob.N])

# p = -sum([sumHinv*H[k]*gt.R*prob.y[:,k] for k = 1:prob.N])
# sum([(gt.R*prob.y[:,i] + p)'*H[i]*(gt.R*prob.y[:,i] + p) for i = 1:prob.N])