## Try to rewrite PACE as a root-finding problem, which should have a closed form solution.

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
    θ = rand()
    ω = randn(3,1); ω /= norm(ω);
    R = axang2rotm(ω, θ)
    # save
    gt = Solution(c, p, R)

    # convert to measurements
    shape = reshape(B*c, (3,N))
    y = R*shape .+ p .+ σm*randn(3,N)

    # save
    return prob, gt, y
end

### MATRIX POLYNOMIAL ATTEMPT
## problem data
prob, gt, y = genproblem(σm=0.)
lam = 0.
K = prob.K
N = prob.N
# still need to add in weights

## eliminate position
ybar = mean(eachcol(y))
Bbar = mean(eachslice(prob.B,dims=2))
# verify (with σm=0.)
# p = ybar - gt.R*Bbar*gt.c

## eliminate shape
Bc = eachslice(prob.B,dims=2) .- [Bbar]
yc = reduce(hcat, eachcol(y) .- [ybar])
A = 2*(sum([Bc[i]'*Bc[i] for i = 1:N]) + lam*I)
invA = inv(A)
# Schur complement
c1 = 2*(invA - invA*ones(K)*inv(ones(K)'*invA*ones(K))*ones(K)'*invA)
c2 = invA*ones(K)*inv(ones(K)'*invA*ones(K))
# verify (with σm=0.)
# c = c1*sum([Bc[i]'*gt.R'*yc[:,i] for i = 1:prob.N]) + c2

# ## Objective (before first order conditions, for verification)
# L = 0.
# # constant terms (will disappear under derivative)
# L += sum([yc[:,i]'*yc[:,i] for i = 1:N])
# L += sum([c2'*Bc[i]'*Bc[i]*c2 for i = 1:N])
# L += lam*(c2'*c2)
# # linear in R => quadratic in q
# L += -2*sum([yc[:,i]'*gt.R*Bc[i] for i = 1:N])*c2 # 1
# L += 2*c2'*sum([Bc[i]'*Bc[i] for i = 1:N])*c1*sum([Bc[j]'*gt.R'*yc[:,j] for j = 1:N]) # 2
# L += lam*2*c2'*c1*sum([Bc[j]'*gt.R'*yc[:,j] for j = 1:N]) # 3
# # quadratic in R => quartic in q
# L += lam*sum([Bc[j]'*gt.R'*yc[:,j] for j = 1:N])'*c1'*c1*sum([Bc[j]'*gt.R'*yc[:,j] for j = 1:N]) # 4
# L += -2*sum([yc[:,i]'*gt.R*Bc[i] for i = 1:N])*c1*sum([Bc[j]'*gt.R'*yc[:,j] for j = 1:N]) # 5
# L += sum([Bc[j]'*gt.R'*yc[:,j] for j = 1:N])'*c1'*sum([Bc[i]'*Bc[i] for i = 1:N])'*c1*sum([Bc[j]'*gt.R'*yc[:,j] for j = 1:N]) # 6

# quaternion version
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

q_gt = rotm2quat(gt.R)

# # objective in quaternion form
# L1 = -2*q_gt'*sum([Ω1(yc[:,i])'*Ω2(Bc[i]*c2) for i = 1:N])*q_gt
# L1_check = -2*sum([yc[:,i]'*gt.R*Bc[i]*c2 for i = 1:N])

# L2 = 2*q_gt'*sum([Ω1(yc[:,j])'*Ω2(Bc[j]*c1'*sum([Bc[i]'*Bc[i] for i = 1:N])*c2) for j = 1:N])*q_gt
# L2_check = 2*c2'*sum([Bc[i]'*Bc[i] for i = 1:N])*c1*sum([Bc[j]'*gt.R'*yc[:,j] for j = 1:N]) # 2

# L3 = lam*2*q_gt'*sum([Ω1(yc[:,j])'*Ω2(Bc[j]*c1'*c2) for j = 1:N])*q_gt
# L3_check = lam*2*c2'*c1*sum([Bc[j]'*gt.R'*yc[:,j] for j = 1:N]) # 3

# # this one is symmetric
# L4_1 = lam*q_gt'*sum([Ω1(yc[:,i])'*Ω2(Bc[i]*c1'*c1*sum([Bc[j]'*gt.R'*yc[:,j] for j = 1:N])) for i = 1:N])*q_gt
# L4_2 = lam*q_gt'*sum([Ω1(yc[:,i])'*Ω2(Bc[i]*c1'*c1*sum([Bc[j]'*gt.R'*yc[:,j] for j = 1:N])) for i = 1:N])*q_gt
# L4_check = lam*sum([Bc[j]'*gt.R'*yc[:,j] for j = 1:N])'*c1'*c1*sum([Bc[j]'*gt.R'*yc[:,j] for j = 1:N]) # 4

# L5_1 = -2*q_gt'*sum([Ω1(yc[:,i])'*Ω2(Bc[i]*c1*sum([Bc[j]'*gt.R'*yc[:,j] for j = 1:N])) for i = 1:N])*q_gt
# L5_2 = -2*q_gt'*sum([Ω1(yc[:,j])'*Ω2(Bc[j]*c1'*sum([Bc[i]'*gt.R'*yc[:,i] for i = 1:N])) for j = 1:N])*q_gt
# L5_check = -2*sum([yc[:,i]'*gt.R*Bc[i] for i = 1:N])*c1*sum([Bc[j]'*gt.R'*yc[:,j] for j = 1:N]) # 5

# # also symmetric
# L6_1 = q_gt'*sum(Ω1(yc[:,k])'*Ω2(Bc[k]*c1'*sum([Bc[i]'*Bc[i] for i = 1:N])'*c1*sum([Bc[j]'*gt.R'*yc[:,j] for j = 1:N])) for k = 1:N)*q_gt
# L6_2 = q_gt'*sum(Ω1(yc[:,k])'*Ω2(Bc[k]*c1'*sum([Bc[i]'*Bc[i] for i = 1:N])'*c1*sum([Bc[j]'*gt.R'*yc[:,j] for j = 1:N])) for k = 1:N)*q_gt
# L6_check = sum([Bc[j]'*gt.R'*yc[:,j] for j = 1:N])'*c1'*sum([Bc[i]'*Bc[i] for i = 1:N])'*c1*sum([Bc[j]'*gt.R'*yc[:,j] for j = 1:N]) # 6

# next steps: 
# 2. take derivatives, verify gt satisfies first order conditions
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
    mat = ℒ(q_guess)
    q_new = normalize(mat*q_guess)
    if abs(abs(q_guess'*q_new) - 1) < 1e-6
        println("Converged in $i iterations.")
        break
    end

    q_guess = q_new
end

# compare:
mat_guess = ℒ(q_guess)
mat_gt = ℒ(q_gt)
@printf "min. eigvals (guess, gt): %.2f, %.2f\n" eigvals(mat_guess)[1] eigvals(mat_gt)[1]
@printf "max. eigvals (guess, gt): %.2f, %.2f\n" eigvals(mat_guess)[end] eigvals(mat_gt)[end]
@printf "quat dif norm: %.3f\n" min(norm(q_guess - q_gt), norm(-q_guess - q_gt))


## Approach 2: Self-Consistent Field Iteration
q_guess2 = normalize(randn(4))
for i = 1:Int(1e4)
    global q_guess2
    mat = ℒ(q_guess2)
    q_new = eigvecs(mat)[:,1]
    normalize!(q_new)
    if abs(abs(q_guess2'*q_new) - 1) < 1e-6
        println("Converged in $i iterations.")
        break
    end

    q_guess2 = q_new
end
# can we use SCF to solve for different eigvecs? Yes! Pretty easily.


# ## Symbolic approach
# using Symbolics

# @variables q[1:4]

# # params
# weights = ones(prob.N)
# lam = 0.0
# K = prob.K

# # define map q -> R
# R = [q[1]^2 + q[2]^2 - q[3]^2 - q[4]^2   2*(q[2]*q[3] - q[1]*q[4])   2*(q[2]*q[4] + q[1]*q[3]);
#      2*(q[2]*q[3] + q[1]*q[4])   q[1]^2 - q[2]^2 + q[3]^2 - q[4]^2   2*(q[3]*q[4] - q[1]*q[2]);
#      2*(q[2]*q[4] - q[1]*q[3])   2*(q[3]*q[4] + q[1]*q[2])   q[1]^2 - q[2]^2 - q[3]^2 + q[4]^2]

# # eliminate c
# Bbar = sum([prob.B[3*(i-1)+1:3*i,:] for i in 1:prob.N])/prob.N
# ybar = sum([y[:,i] for i in 1:prob.N])/prob.N
# A = 2*sum([(prob.B[3*(i-1)+1:3*i,:] - Bbar)'*(prob.B[3*(i-1)+1:3*i,:] - Bbar) for i in 1:prob.N]) + 2*lam*I
# Ainv = inv(A)
# c = (Ainv - Ainv*ones(K)*inv(ones(K)'*Ainv*ones(K))*ones(K)'*Ainv)*(2*sum([(prob.B[3*(i-1)+1:3*i,:] - Bbar)'*R'*(y[:,i] - ybar) for i in 1:prob.N])) + Ainv*ones(K)*inv(ones(K)'*Ainv*ones(K))

# # eliminate p
# p = sum([y[:,i] - R*prob.B[3*(i-1)+1:3*i,:]*c])/prob.N

# # PACE objective (p, c eliminated)
# obj = sum([(y[:,i] - R*prob.B[3*(i-1)+1:3*i,:]*c - p)'*(y[:,i] - R*prob.B[3*(i-1)+1:3*i,:]*c - p) for i in 1:prob.N])
# obj += lam*(c'*c)




# Temporarily in function to preserve
function powermethod()
    ### Power method
    # 1) pick random q
    # 2) q_new = A * q (normalize)
    # 3) normalize q_new

    A = randn(4,4)
    B = randn(4,4)

    qs = []
    evs = []
    for j = 1:10
        q = normalize(randn(4))
        for i = 1:Int(1e4)
            mat = A*(q'*q)*B
            q_new = normalize(mat*q)
            if abs(abs(q'*q_new) - 1) < 1e-6
                println("Converged in $i iterations.")
                break
            end

            q = q_new
        end
        push!(qs, q)
        mat = A*(q'*q)*B
        append!(evs, eigvals(mat)[end])
    end

    qs = reduce(hcat, qs)

    # check
    q = qs[:,1]
    mat = A*(q'*q)*B
    [q eigvecs(mat)[:,end]]
end