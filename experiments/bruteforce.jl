## Brute force a solution using only GNC for outlier rejection
# run simulation first I guess
# TODO: move to function

# Include solvers (TODO: probably should improve this)
include("../src/pace_tools.jl")
include("../src/gnc.jl")

# define problem
σm = 0.04
prob, gt, y = genproblem(N=5, K=2, σm=σm)

# add bad associations
B_new = repeat(prob.B, inner=[1,prob.N,1])
y_new = repeat(y, 1, prob.N)
prob_new = Problem(prob.N^2, prob.K, prob.σm, prob.r, B_new)

# solve with GNC!
function scf_for_gnc(prob, y, weights)
    # soln, cost, residuals = solver(prob, y, weights)
    lam = 0.0
    soln, obj_val = solvePACE_SCF(prob, y, weights, lam; global_iters=2)

    B = reshape(prob.B, 3*prob.N, prob.K)
    shape = reshape(B*soln.c, (3,prob.N))
    residuals = norm.(eachcol(y - (soln.R*shape .+ soln.p))).^2

    return soln, obj_val, residuals
end

soln, inliers = gnc(prob_new, y_new, scf_for_gnc; cbar2=0.008)

_, R_err_scf = rotm2axang(gt.R'*soln.R); R_err_scf *= 180/π
println(R_err_scf)
inliers

# TODO: runtime histogram of each method + GNC
function gendatas(num)
    datas = []
    σm = 0.04
    for _ = 1:num
        prob, gt, y = genproblem(N=5, K=2, σm=σm)

        B_new = repeat(prob.B, inner=[1,prob.N,1])
        y_new = repeat(y, 1, prob.N)
        prob_new = Problem(prob.N^2, prob.K, prob.σm, prob.r, B_new)
        push!(datas, (prob_new, gt, y_new))
    end
    return datas
end
datas = gendatas(1000)


function sdp_for_gnc(prob, y, weights)
    # soln, cost, residuals = solver(prob, y, weights)
    lam = 0.0
    soln, obj_val = solvePACE_TSSOS(prob, y, weights, lam)

    B = reshape(prob.B, 3*prob.N, prob.K)
    shape = reshape(B*soln.c, (3,prob.N))
    residuals = norm.(eachcol(y - (soln.R*shape .+ soln.p))).^2

    return soln, obj_val, residuals
end
function manopt_for_gnc(prob, y, weights)
    # soln, cost, residuals = solver(prob, y, weights)
    lam = 0.0
    soln, obj_val = solvePACE_Manopt(prob, y, weights, lam)

    B = reshape(prob.B, 3*prob.N, prob.K)
    shape = reshape(B*soln.c, (3,prob.N))
    residuals = norm.(eachcol(y - (soln.R*shape .+ soln.p))).^2

    return soln, obj_val, residuals
end

methods = [scf_for_gnc, sdp_for_gnc, manopt_for_gnc]

# success rate
success_count = zeros(3)
for (idx, data) in enumerate(datas)
    for (i, method) in enumerate(methods)
        soln, inliers = gnc(data[1], data[3], method; cbar2=0.008)
        _, R_err = rotm2axang(gt.R'*soln.R); R_err *= 180/π
        if R_err < 10.
            success_count[i] += 1
        end
    end
    println(idx)
end