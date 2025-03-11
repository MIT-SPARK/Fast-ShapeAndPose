## Compare PACE with "local" solver

using Random
using Statistics
using LinearAlgebra
using JuMP

using TSSOS
using DynamicPolynomials
using Printf

using LorenzoRotations

struct Problem
    N ::Int64       # num keypoints
    K ::Int64       # num shapes
    σm::Float64     # measurement noise standard deviation
    r ::Float64     # shape radius
    B ::Matrix{Float64} # shape library
end

struct Solution
    c ::Matrix{Float64} # gt shape coefficent
    p ::Matrix{Float64} # gt position
    R ::Matrix{Float64} # gt rotation
end

function genproblem(;N::Int64=7, K::Int64=4, σm::Float64=0.05, r::Float64=0.2)
    # generate shape lib
    meanShape = randn(3, N)
    meanShape .-= mean(meanShape,dims=2)
    
    shapes = zeros(3,N,K)
    for k = 1:K
        shapes[:,:,k] = meanShape + r * randn(3,N)
    end
    B = reshape(shapes, 3*N, K)
    prob = Problem(N, K, σm, r, B)

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

### SOLVE VIA TSSOS
function solvePACE_TSSOS(prob, y, weights, lam=0.1)
    @polyvar R[1:3,1:3]
    vars = vec(R)

    # symbolic expressions for c, p
    W = Diagonal(repeat(weights, inner=3))
    yw = zeros(3,1)
    Bw = zeros(3,prob.K)
    for i = 1:prob.N
        yw += weights[i]*y[:,i]
        Bw += weights[i]*prob.B[(1+3*(i-1)):(3*i), :]
    end
    yw /= sum(weights)
    Bw /= sum(weights)
    Bbar = prob.B - repeat(Bw, outer=prob.N)
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
        term = R'*(y[:,i] - yw) - (prob.B[1+3*(i-1):3*i,:] - Bw)*c
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
    stime = @elapsed begin
        opt, sol, data = tssos_first(pop, vars, 1, numeq=length(eq), solve=true, quotient=false, QUIET=true)
    end

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

    return soln, gap, stime
end

import Manifolds
import Manopt
### SOLVE VIA JuMP
function solvePACE_JuMP(prob, y, weights, lam=0.1)
    SO3 = Manifolds.Rotations(3)

    model = Model()
    @variable(model, R[1:3,1:3] in SO3)
    set_start_value.(R, diagm(ones(3)))
    set_optimizer(model, Manopt.JuMP_Optimizer)

    # symbolic expressions for c, p
    W = Diagonal(repeat(weights, inner=3))
    yw = zeros(3,1)
    Bw = zeros(3,prob.K)
    for i = 1:prob.N
        yw += weights[i]*y[:,i]
        Bw += weights[i]*prob.B[(1+3*(i-1)):(3*i), :]
    end
    yw /= sum(weights)
    Bw /= sum(weights)
    Bbar = prob.B - repeat(Bw, outer=prob.N)
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
        term = R'*(y[:,i] - yw) - (prob.B[1+3*(i-1):3*i,:] - Bw)*c
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
    stime = @elapsed optimize!(model)

    # Convert to solution
    R_est = value.(R)
    R_est = project2SO3(R_est)
    c_est = value.(c)
    p_est = value.(p)

    soln = Solution(c_est, p_est, R_est)

    return soln, stime
end

### Compare!
prob, gt, y = genproblem(σm=1.0)
soln_tssos, gap_tssos, stime_tssos = solvePACE_TSSOS(prob, y, ones(prob.N))
soln_jump, stime_jump = solvePACE_JuMP(prob, y, ones(prob.N))

# errors
_, R_err_tssos = rotm2axang(gt.R'*soln_tssos.R)
R_err_tssos *= 180/π
_, R_err_jump = rotm2axang(gt.R'*soln_jump.R)
R_err_jump *= 180/π

if gap_tssos > 1e-6
    @printf "LOST TIGHTNESS!\n"
end

@printf "-------------------\n"
@printf "Solve ms: %.2f (TSSOS), %.2f (JuMP)\n" stime_tssos*1000 stime_jump*1000
@printf "Deg errs: %.2f (TSSOS), %.2f (JuMP)\n" R_err_tssos R_err_jump