## SCF for pnp
# Lorenzo Shaikewitz, 4/4/2025

include("../src/pnp_tools.jl")

prob, gt, u = genPnPproblem(σm = 0., r=0.8)
weights = ones(prob.N)

H = [weights[i]*Symmetric((u[:,i]*[0 0 1] - I)'*(u[:,i]*[0 0 1] - I)) for i = 1:prob.N]
sumHinv = inv(sum(H))

## Lagrangian terms
function Lquartic(q)
    R = quat2rotm(q)
    L1 = sum([-Ω1(H[i]*R*prob.y[:,i])*Ω2(prob.y[:,i]) for i = 1:prob.N])
    L2 = sum([sum([-Ω1(H[k]*sumHinv*H[i]*sumHinv*sum([H[j]*R*prob.y[:,j] for j = 1:prob.N]))*Ω2(prob.y[:,k]) for k = 1:prob.N]) for i = 1:prob.N])
    L3 = sum([sum([Ω1(2*H[k]*sumHinv*H[i]*R*prob.y[:,i])*Ω2(prob.y[:,k]) for k = 1:prob.N]) for i = 1:prob.N])
    return 4*(L1 + L2 + L3)
end
# Lagrangian
ℒ(q) = Symmetric(Lquartic(q))

## SCF
q_logs = []
mat_logs = []
# intial guess
q_scf = normalize(randn(4))
for i = 1:100
    global q_scf
    # update
    local mat = ℒ(q_scf)
    q_scf = eigvecs(mat)[:,1]
    push!(q_logs, q_scf)
    push!(mat_logs, mat)
end

# for PnP, we can find the max really easily but not the min
# for 0 noise the min cost is exactly 0...

# test other solution
