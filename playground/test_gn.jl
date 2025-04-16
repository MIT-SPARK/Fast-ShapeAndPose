# Test Gauss-Newton

include("../src/pace_tools.jl")

prob, gt, y = genproblem(σm=0.)

weights = ones(prob.N)
lam = 0.
λ = 0.

R₀ = randrotation()

if sum(weights .!= 0) <= prob.K
    lam = 0.01
end

# symbolic expressions for c, p
# B = reshape(prob.B, 3*prob.N, prob.K)

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
    c = G*(2*lam*cbar + 2*sum([Btild[i]'*R'*ytild[i] for i = 1:prob.N])) + g
end
# optimal position given shape & rotation
p(R,c) = yw - R*Bw*c

# registration residual
ri(R, i) = R'*ytild[i] - Btild[i]*c(R)
# shape residual
rc(R) = √lam*(c(R) - cbar)

# TODO: fix Jacobians, residuals, and remove unnecessary vars

# Registration Jacobian
Ji(R, i) = R'*skew(ytild[i]) - 2*Btild[i]*G*sum([Btild[j]'*R'*skew(ytild[j]) for j = 1:prob.N])

# Shape jacobian
Jc(R) = 2*√lam*G*sum([Btild[i]'*R'*skew(ytild[i]) for i = 1:prob.N])

# G-N / L-M
R_cur = R₀
for i = 1:100
    global R_cur
    Σ = Jc(R_cur)'*Jc(R_cur) + sum([Ji(R_cur,i)'*Ji(R_cur,i) for i = 1:prob.N])
    v = Jc(R_cur)'*rc(R_cur) + sum([Ji(R_cur,i)'*ri(R_cur,i) for i = 1:prob.N])
    δθ = -inv(Σ + λ*I)*v
    α  = 1.0
    R_cur = exp(skew(α*δθ))*R_cur
    println(norm(δθ))
    display(R_cur)

    # check for convergence
    if norm(δθ) < 1e-3
        println("Convergence?")
        break
    end
end