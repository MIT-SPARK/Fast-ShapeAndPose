## Compare local solutions to Manopt with same initial guess
# Lorenzo Shaikewitz, 3/31/2025

include("../src/pace_tools.jl")

# generate problem
Random.seed!(14)

two = 0
mismatched = 0

for i = 1:1000
    global two, mismatched
    prob, gt, y = genproblem(σm=1.)
    lam = 0.0
    weights = ones(prob.N)

    q0 = normalize(randn(4))

    # solve via SCF with logs
    soln, obj_val, q_logs, ℒ, obj = solvePACE_SCF(prob, y, weights, lam; q0=q0, global_iters=15, debug=true)

    # solve via Manopt
    soln_manopt, obj_manopt = solvePACE_Manopt(prob, y, weights, lam; R0=quat2rotm(q0))

    q_scf = rotm2quat(soln.R)
    q_man = rotm2quat(soln_manopt.R)

    if length(q_logs) > 1
        two += 1
        @printf "%d solutions. Obj gap: %.1e\n" length(q_logs) (obj_val - obj_manopt)
        if abs(abs(q_scf'*q_man) - 1) > 1e-5
            printstyled("SCF does not match manifold\n", color=:red)
            mismatched += 1
        end
    end
end

@printf "Mismatched: %.1f%%\n" (mismatched / two * 100)