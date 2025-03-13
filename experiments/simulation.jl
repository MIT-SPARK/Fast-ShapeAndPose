# Solve simulated PACE problems with quaternion and SDP solvers.
# Goal: produce quantitative results about tightness/accuracy.

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
function solvePACE_TSSOS(prob, y, weights, lam=0.)
    @polyvar R[1:3,1:3]
    vars = vec(R)

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
        obj += term' * term
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

    return soln, gap, opt
end

### MANOPT (LOCAL) SOLVER
using JuMP
import Manifolds
import Manopt
### SOLVE VIA JuMP
function solvePACE_Manopt(prob, y, weights, lam=0.)
    SO3 = Manifolds.Rotations(3)

    model = Model()
    @variable(model, R[1:3,1:3] in SO3)
    set_start_value.(R, diagm(ones(3)))
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
        obj += term' * term
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

# weights not currently working
function solvePACE_scf(prob, y, weights, lam=0.; grid=false)
    K = prob.K
    N = prob.N
    # still need to add in weights

    ## eliminate position
    ybar = mean(eachcol(y))
    Bbar = mean(eachslice(prob.B,dims=2))

    ## eliminate shape
    Bc = eachslice(prob.B,dims=2) .- [Bbar]
    yc = reduce(hcat, eachcol(y) .- [ybar])
    A = 2*(sum([Bc[i]'*Bc[i] for i = 1:N]) + lam*I)
    invA = inv(A)
    # Schur complement
    c1 = 2*(invA - invA*ones(K)*inv(ones(K)'*invA*ones(K))*ones(K)'*invA)
    c2 = invA*ones(K)*inv(ones(K)'*invA*ones(K))

    """
    Lagrangian derivative without `μ(1-q'*q)`
    """
    function ℒ(q) # \scrL
        # linear in q
        L1  = -2*sum([Ω1(yc[:,i])'*Ω2(Bc[i]*c2) for i = 1:N])
        L1 += L1'
        L2  =  2*sum([Ω1(yc[:,j])'*Ω2(Bc[j]*c1'*sum([Bc[i]'*Bc[i] for i = 1:N])*c2) for j = 1:N])
        L2 +=  L2'
        L3  =  2*lam*sum([Ω1(yc[:,j])'*Ω2(Bc[j]*c1'*c2) for j = 1:N])
        L3 +=  L3'
        # quadratic in q
        R = quat2rotm(q)
        L4  = 2*lam*sum([Ω1(yc[:,i])'*Ω2(Bc[i]*c1'*c1*sum([Bc[j]'*R'*yc[:,j] for j = 1:N])) for i = 1:N])
        L4 += L4'
        L5  = -2*sum([Ω1(yc[:,i])'*Ω2(Bc[i]*c1*sum([Bc[j]'*R'*yc[:,j] for j = 1:N])) for i = 1:N])
        L5 += -2*sum([Ω1(yc[:,j])'*Ω2(Bc[j]*c1'*sum([Bc[i]'*R'*yc[:,i] for i = 1:N])) for j = 1:N])
        L5 += L5'
        L6  = 2*sum(Ω1(yc[:,k])'*Ω2(Bc[k]*c1'*sum([Bc[i]'*Bc[i] for i = 1:N])'*c1*sum([Bc[j]'*R'*yc[:,j] for j = 1:N])) for k = 1:N)
        L6 += L6'
        return L1 + L2 + L3 + (L4 + L5 + L6)
        # works better in high noise:
        # return L1 + L2 + L3 + 1/2*(L4 + L5 + L6)
    end
    """
    Full objective value.
    """
    function ℒ2(q) # \scrL
        L = sum([yc[:,i]'*yc[:,i] for i = 1:N])
        L += sum([c2'*Bc[i]'*Bc[i]*c2 for i = 1:N])
        L += lam*(c2'*c2)
        # constant terms
        L1  = -2*sum([Ω1(yc[:,i])'*Ω2(Bc[i]*c2) for i = 1:N])
        L1 += -2*sum([Ω1(yc[:,i])'*Ω2(Bc[i]*c2) for i = 1:N])'
        L2  =  2*sum([Ω1(yc[:,j])'*Ω2(Bc[j]*c1'*sum([Bc[i]'*Bc[i] for i = 1:N])*c2) for j = 1:N])
        L2 +=  2*sum([Ω1(yc[:,j])'*Ω2(Bc[j]*c1'*sum([Bc[i]'*Bc[i] for i = 1:N])*c2) for j = 1:N])'
        L3  =  2*lam*sum([Ω1(yc[:,j])'*Ω2(Bc[j]*c1'*c2) for j = 1:N])
        L3 +=  2*lam*sum([Ω1(yc[:,j])'*Ω2(Bc[j]*c1'*c2) for j = 1:N])'
        # quadratic in q
        R = quat2rotm(q)
        L4  = lam*sum([Ω1(yc[:,i])'*Ω2(Bc[i]*c1'*c1*sum([Bc[j]'*R'*yc[:,j] for j = 1:N])) for i = 1:N])
        L4 += lam*sum([Ω1(yc[:,i])'*Ω2(Bc[i]*c1'*c1*sum([Bc[j]'*R'*yc[:,j] for j = 1:N])) for i = 1:N])'
        L4 += lam*sum([Ω1(yc[:,i])'*Ω2(Bc[i]*c1'*c1*sum([Bc[j]'*R'*yc[:,j] for j = 1:N])) for i = 1:N])
        L4 += lam*sum([Ω1(yc[:,i])'*Ω2(Bc[i]*c1'*c1*sum([Bc[j]'*R'*yc[:,j] for j = 1:N])) for i = 1:N])'
        L5  = -2*sum([Ω1(yc[:,i])'*Ω2(Bc[i]*c1*sum([Bc[j]'*R'*yc[:,j] for j = 1:N])) for i = 1:N])
        L5 += -2*sum([Ω1(yc[:,i])'*Ω2(Bc[i]*c1*sum([Bc[j]'*R'*yc[:,j] for j = 1:N])) for i = 1:N])'
        L5 += -2*sum([Ω1(yc[:,j])'*Ω2(Bc[j]*c1'*sum([Bc[i]'*R'*yc[:,i] for i = 1:N])) for j = 1:N])
        L5 += -2*sum([Ω1(yc[:,j])'*Ω2(Bc[j]*c1'*sum([Bc[i]'*R'*yc[:,i] for i = 1:N])) for j = 1:N])'
        L6  = sum(Ω1(yc[:,k])'*Ω2(Bc[k]*c1'*sum([Bc[i]'*Bc[i] for i = 1:N])'*c1*sum([Bc[j]'*R'*yc[:,j] for j = 1:N])) for k = 1:N)
        L6 += sum(Ω1(yc[:,k])'*Ω2(Bc[k]*c1'*sum([Bc[i]'*Bc[i] for i = 1:N])'*c1*sum([Bc[j]'*R'*yc[:,j] for j = 1:N])) for k = 1:N)'
        L6 += sum(Ω1(yc[:,k])'*Ω2(Bc[k]*c1'*sum([Bc[i]'*Bc[i] for i = 1:N])'*c1*sum([Bc[j]'*R'*yc[:,j] for j = 1:N])) for k = 1:N)
        L6 += sum(Ω1(yc[:,k])'*Ω2(Bc[k]*c1'*sum([Bc[i]'*Bc[i] for i = 1:N])'*c1*sum([Bc[j]'*R'*yc[:,j] for j = 1:N])) for k = 1:N)'
        return q'*(1/2*(L1 + L2 + L3) + 1/4*(L4 + L5 + L6))*q + L
    end

    ## Optimal solution:
    q_scf = zeros(4)
    obj = 100.
    # repeat just to make sure
    for i = 1:(if grid 1 else 10 end)
        # look for the eigenvector q corresponding to the MINIMUM eigenvalue of ℒ(q)
        q_guess = normalize(randn(4))
        for _ = 1:Int(1e3)
            mat = ℒ(q_guess)
            q_new = eigvecs(mat)[:,1]; normalize!(q_new); # SCF
            # q_new = normalize(mat*q_guess) # Power iteration
            if abs(abs(q_guess'*q_new) - 1) < 1e-8
                q_guess = q_new
                break
            end
            q_guess = q_new
        end
        obj_mat = ℒ2(q_guess)
        if obj_mat < obj
            obj = obj_mat
            q_scf = q_guess
        end
    end

    # Convert to solution
    R_est = quat2rotm(q_scf)
    c_est = reshape(c1*sum([Bc[i]'*R_est'*yc[:,i] for i = 1:prob.N]) + c2,prob.K,1)
    p_est = ybar - R_est*Bbar*c_est

    soln = Solution(c_est, p_est, R_est)

    obj = ℒ2(q_scf)

    # for grid
    qs = [q_scf]
    if grid
        # search for other maximum eigenvectors!
        for i = 1:2000
            q_guess = normalize(randn(4))
            for i = 1:100
                mat = ℒ(q_guess)
                q_new = eigvecs(mat)[:,1]

                # stop if too close to q_scf
                extrabreak = false
                for quat in qs
                    if abs(abs(q_new'*quat) - 1) < 1e-5
                        extrabreak = true
                        break
                    end
                end
                if extrabreak
                    break
                end

                # stop if constant
                if abs(abs(q_guess'*q_new) - 1) < 1e-8
                    push!(qs, q_guess)
                    break
                end

                q_guess = q_new
            end
        end
    end
    
    return soln, ℒ, obj, qs
end

### Simulate!
σm = 1.0

repeats = 10000
gaps = zeros(repeats)
R_errs = zeros(repeats,3)
p_errs = zeros(repeats,3)
c_errs = zeros(repeats,3)
objs = zeros(repeats,3)

datas = []

for i = 1:repeats
    # Define problem
    prob, gt, y = genproblem(σm=σm)
    lam = 0.0
    weights = ones(prob.N)

    # Solve
    soln_tssos, gap_tssos, obj_tssos = solvePACE_TSSOS(prob, y, weights, lam)
    soln_manopt, obj_manopt = solvePACE_Manopt(prob, y, weights, lam)
    soln_scf, ℒ, obj_scf, qs = solvePACE_scf(prob, y, weights, lam; grid=(gap_tssos > 1e-5))

    # Compute Errors
    _, R_err_tssos = rotm2axang(gt.R'*soln_tssos.R); R_err_tssos *= 180/π
    _, R_err_manopt = rotm2axang(gt.R'*soln_manopt.R); R_err_manopt *= 180/π
    _, R_err_scf = rotm2axang(gt.R'*soln_scf.R); R_err_scf *= 180/π

    p_errs[i,1] = norm(soln_tssos.p - gt.p)
    p_errs[i,2] = norm(soln_manopt.p - gt.p)
    p_errs[i,3] = norm(soln_scf.p - gt.p)
    c_errs[i,1] = norm(soln_tssos.c - gt.c)
    c_errs[i,2] = norm(soln_manopt.c - gt.c)
    c_errs[i,3] = norm(soln_scf.c - gt.c)

    # if R_err_scf - R_err_tssos > 10
    #     global data = (R_err_scf, R_err_tssos, ℒ, rotm2quat(soln_tssos.R), rotm2quat(soln_scf.R), prob, gt, y)
    #     break
    # end

    # Save
    gaps[i] = gap_tssos
    R_errs[i,1] = R_err_tssos
    R_errs[i,2] = R_err_manopt
    R_errs[i,3] = R_err_scf
    objs[i,1] = obj_tssos
    objs[i,2] = obj_manopt
    objs[i,3] = obj_scf
    push!(datas, (ℒ, rotm2quat(soln_tssos.R), rotm2quat(soln_manopt.R), rotm2quat(soln_scf.R), prob, gt, y, qs))
    if i % 10 == 0
        print("$i ")
    end
end

### Plot!
import Plots

p1 = Plots.scatter(abs.(gaps), p_errs[:,2:3] .- p_errs[:,1], label=["Local" "SCF"], title="Difference from SDP")
Plots.plot!(xscale=:log10, legend=:bottom, xlabel="Suboptimality Gap", ylabel="Pos Error (m)")

p2 = Plots.scatter(abs.(gaps), c_errs[:,2:3] .- c_errs[:,1], label=["Local" "SCF"], title="Difference from SDP")
Plots.plot!(xscale=:log10, legend=:bottom, xlabel="Suboptimality Gap", ylabel="Shape Error")

p3 = Plots.scatter(abs.(gaps), R_errs[:,2:3] .- R_errs[:,1], label=["Local" "SCF"], title="Difference from SDP")
Plots.plot!(xscale=:log10, legend=:bottom, xlabel="Suboptimality Gap", ylabel="Rot Error (deg)")

### debugging
# q_ts = data[4]
# q_scf = data[5]