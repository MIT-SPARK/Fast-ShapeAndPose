## Brute force a solution using only GNC for outlier rejection
# TODO: move to function

# Include solvers (TODO: probably should improve this)
include("../src/pace_tools.jl")
include("../src/gnc.jl")

# define problem
# σm = 0.02
# prob, gt, y = genproblem(N=5, K=2, σm=σm)

# # add bad associations
# B_new = repeat(prob.B, inner=[1,prob.N,1])
# y_new = repeat(y, 1, prob.N)
# prob_new = Problem(prob.N^2, prob.K, prob.σm, prob.r, B_new)

# solve with GNC!
function scf_for_gnc(prob, y, weights)
    # soln, cost, residuals = solver(prob, y, weights)
    lam = 0.01
    soln, obj_val = solvePACE_SCF(prob, y, weights, lam; global_iters=1)

    B = reshape(prob.B, 3*prob.N, prob.K)
    shape = reshape(B*soln.c, (3,prob.N))
    residuals = norm.(eachcol(y - (soln.R*shape .+ soln.p))).^2

    return soln, obj_val, residuals
end
function sdp_for_gnc(prob, y, weights)
    # soln, cost, residuals = solver(prob, y, weights)
    lam = 0.01
    soln, obj_val, _ = solvePACE_TSSOS(prob, y, weights, lam)

    B = reshape(prob.B, 3*prob.N, prob.K)
    shape = reshape(B*soln.c, (3,prob.N))
    residuals = norm.(eachcol(y - (soln.R*shape .+ soln.p))).^2

    return soln, obj_val, residuals
end
function manopt_for_gnc(prob, y, weights)
    # soln, cost, residuals = solver(prob, y, weights)
    lam = 0.01
    soln, obj_val = solvePACE_Manopt(prob, y, weights, lam)

    B = reshape(prob.B, 3*prob.N, prob.K)
    shape = reshape(B*soln.c, (3,prob.N))
    residuals = norm.(eachcol(y - (soln.R*shape .+ soln.p))).^2

    return soln, obj_val, residuals
end
methods = [sdp_for_gnc, manopt_for_gnc, scf_for_gnc]

# soln, inliers, success = gnc(prob_new, y_new, scf_for_gnc; cbar2=0.008)

# _, R_err_scf = rotm2axang(gt.R'*soln.R); R_err_scf *= 180/π
# println(R_err_scf)
# inliers

# TODO: runtime histogram of each method + GNC
function gendatas(num)
    datas = []
    σm = 0.04
    for _ = 1:num
        prob, gt, y = genproblem(N=5, K=2, σm=σm)

        # just first one gets all correspondences
        B_new = prob.B[:,[1;2:end;2:end],:]
        y_new = y[:,[1;Int.(ones(prob.N-1));2:end]]
        prob_new = Problem(2*prob.N-1, prob.K, prob.σm, prob.r, B_new)

        # all correspondences
        # B_new = repeat(prob.B, inner=[1,prob.N,1])
        # y_new = repeat(y, 1, prob.N)
        # prob_new = Problem(prob.N^2, prob.K, prob.σm, prob.r, B_new)
        push!(datas, (prob_new, gt, y_new))
    end
    return datas
end
num_datas = 1000
datas = gendatas(num_datas)

# success rate and timing
success_count = zeros(3)
success2_count = zeros(3)
times = zeros(3,num_datas)
for (idx, data) in enumerate(datas)
    for (i, method) in enumerate(methods)
        t = @timed gnc(data[1], data[3], method; cbar2=0.008)
        soln, inliers, success = t.value
        times[i,idx] = t.time
        _, R_err = rotm2axang(data[2].R'*soln.R); R_err *= 180/π
        if R_err < 10.
            success_count[i] += 1
        end
        # if inliers == [1; 7; 13; 19; 25]
        if inliers == [1; 6;7;8;9]
            success2_count[i] += 1
        end
    end
    if idx % 10 == 0
        print("$idx ")
    end
end