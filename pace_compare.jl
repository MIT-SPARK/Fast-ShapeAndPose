# Script for qualitative comparisons of 

using Random
using Statistics
using LinearAlgebra

using Printf

using MathUtils

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
    R = unifrandrotation()
    # save
    gt = Solution(c, p, R)

    # convert to measurements
    shape = reshape(B*c, (3,N))
    y = R*shape .+ p .+ σm*randn(3,N)

    # save
    return prob, gt, y
end

# debugging tool for functions
xx = Ref{Any}()

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
    gap = (obj_est - opt) / (opt+1);

    # @printf "projected optimum = %.4e\n" obj_est
    # @printf "second eigenvalue = %.4e\n" eigvals(mom)[end-1]
    # @printf "stable gap = %.4e\n" gap

    return soln, gap
end

### DEFINE PROBLEM
σm = 1.0
prob, gt, y = genproblem(σm=σm)
lam = 0.0
weights = ones(prob.N)

### SOLVE VIA QUATERNION
## A quick note on speed:
#  The quat form is currently very slow, but this is mostly because of rebuilding matrices.
#  The current implementation is hardly optimized. The main bottleneck is computing ℒ once.
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
function solvePACE_quat(prob, y, weights, lam=0.)
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
    Lagrangian without `μ(1-q'*q)`
    """
    function ℒ(q) # \scrL
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
        return L1 + L2 + L3 + L4 + L5 + L6
    end

    ## Optimal solution:
    # look for the eigenvector q corresponding to the MINIMUM eigenvalue of ℒ(q)
    q_guess = normalize(randn(4))
    for i = 1:Int(1e4)
        mat = ℒ(q_guess)
        q_new = normalize(mat*q_guess)
        if abs(abs(q_guess'*q_new) - 1) < 1e-6
            println("Converged in $i iterations.")
            break
        end

        q_guess = q_new
    end
    normalize!(q_guess)

    # Convert to solution
    R_est = quat2rotm(q_guess)
    c_est = reshape(c1*sum([Bc[i]'*R_est'*yc[:,i] for i = 1:prob.N]) + c2,prob.K,1)
    p_est = ybar - R_est*Bbar*gt.c

    soln = Solution(c_est, p_est, R_est)
    return soln
end
function runquat()
    _, err = rotm2axang(gt.R'*solvePACE_quat(prob, y, weights, lam).R)
    err*=180/π
end

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
Lagrangian without `μ(1-q'*q)`
"""
function ℒ(q) # \scrL
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
    return L1 + L2 + L3 + L4 + L5 + L6
end

## Optimal solution:
# look for the eigenvector q corresponding to the MINIMUM eigenvalue of ℒ(q)
q_guess = normalize(randn(4))
for i = 1:Int(1e4)
    global q_guess
    local mat = ℒ(q_guess)
    q_new = normalize(mat*q_guess)
    if abs(abs(q_guess'*q_new) - 1) < 1e-6
        println("Quaternion converged ($i iterations)")
        break
    end

    q_guess = q_new
end
normalize!(q_guess)

## Approach 2: Self-Consistent Field Iteration
q_guess2 = normalize(randn(4))
for i = 1:Int(1e4)
    global q_guess2
    local mat = ℒ(q_guess2)
    q_new = eigvecs(mat)[:,1]
    normalize!(q_new)
    if abs(abs(q_guess2'*q_new) - 1) < 1e-6
        println("SCF converged ($i iterations).")
        break
    end

    q_guess2 = q_new
end

matqu = ℒ(q_guess)
matscf = ℒ(q_guess2)


# convert to solution
R_quat = quat2rotm(q_guess)
c_quat = reshape(c1*sum([Bc[i]'*R_quat'*yc[:,i] for i = 1:prob.N]) + c2,prob.K,1)
p_quat = ybar - R_quat*Bbar*c_quat
soln_quat = Solution(c_quat, p_quat, R_quat)


### SOLVE VIA TSSOS
soln_tssos, gap_tssos = solvePACE_TSSOS(prob, y, weights, lam)

### COMPARE
q_gt = rotm2quat(gt.R)
q_ts = rotm2quat(soln_tssos.R)
matgt = ℒ(q_gt)
matts = ℒ(q_ts)

# errors
_, R_err_tssos = rotm2axang(gt.R'*soln_tssos.R); R_err_tssos *= 180/π
_, R_err_quat = rotm2axang(gt.R'*soln_quat.R);   R_err_quat  *= 180/π
_, R_err_scf = rotm2axang(gt.R'*quat2rotm(q_guess2)); R_err_scf *= 180/π

@printf "-------------------\n"
# @printf "Solve ms: %.2f (TSSOS), %.2f (quat)\n" stime_tssos*1000 stime_jump*1000
@printf "Deg. errors: %.2f (TSSOS), %.2f (quat), %.2f (scf)\n" R_err_tssos R_err_quat R_err_scf


## Hypothesized failure cases
# 0. tightness
begin
if gap_tssos > 1e-3
    printstyled(@sprintf "Failure 0: lost tightness (%.2e)\n" gap_tssos; color=:red)
end

# 1. `gt.R` close to `I`
if norm(gt.R - I) < 0.4
    printstyled(@sprintf "Failure 1: R near I (%.2f)\n" norm(gt.R - I); color=:red)
    # Breaks: 1
    # Irrelevant: 0
end

# 2. Max. magnitude eigenvalue is positive
ev = eigvals(matqu)
if abs(ev[1]) < ev[end]
    printstyled(@sprintf "Failure 2: |λ_max| > |λ_min| (%.2f > %.2f)\n" ev[end] -ev[1]; color=:red)
    # Breaks: 5
    # Irrelevant: 0
end

# 3. q_gt scalar part close to 1
if abs(q_gt[1]) >0.95
    printstyled(@sprintf "Failure 3: q_gt[1] close to 1 (%.2f)\n" q_gt[1]; color=:red)
    # Breaks: 0
    # Irrelevant: 1
end

## Observations
# 1. TSSOS and quat errors very different
if abs(R_err_quat - R_err_tssos) > 10
    printstyled(@sprintf "Obs. 1: quat, tssos errors different\n"; color=:red)
end

# 2. Two eigvals should be negative
ev_ts = eigvals(matts)
if (ev[2] > 0) || (ev_ts[2] > 0)
    printstyled(@sprintf "Obs. 2: 3+ positive λ (%.2f, %.2f)\n" ev[2] ev_ts[2]; color=:blue)
    # Breaks: 1
    # Irrelevant: 13+
    # this doesn't seem to matter
end

# 3. Quat should be global min
if ev_ts[1] - ev[1] < -1.
    printstyled(@sprintf "Obs. 3: SDP min lower (%.2f < %.2f)\n" ev_ts[1] ev[1]; color=:blue)
end

# 4. TSSOS soln should be eigvec of matts
evec_ts = eigvecs(matts)[:,1]
if min(norm(evec_ts - q_ts), norm(-evec_ts - q_ts)) > 0.01
    printstyled(@sprintf "Obs. 4: SDP min not eigvec\n"; color=:blue)
    # this is loss of tightness, by definition
end

# 5. q_guess should align with q_guess2
if min(norm(q_guess - q_guess2), norm(-q_guess - q_guess2)) > 0.01
    printstyled(@sprintf "Obs. 5: SCF not aligned with power method\n"; color=:blue)
    # this is loss of tightness, by definition
end

end

## TODO: plot!
## TODO: align TSSOS objective with eigvals (should just be removing constant terms)