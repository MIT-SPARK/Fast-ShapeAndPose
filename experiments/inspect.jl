# inspect results of simulation
function scf(ℒ, eigval=1; q0=nothing)
    q_scf = q0
    if isnothing(q0)
        q_scf = normalize(randn(4))
    end
    for i = 1:Int(1e6)
        mat = ℒ(q_scf)
        q_new = eigvecs(mat)[:,eigval]
        normalize!(q_new)
        if abs(abs(q_scf'*q_new) - 1) < 1e-8
            # println("SCF converged ($i iterations).")
            break
        end
        q_scf = q_new
    end
    return q_scf
end

# set criteria
dif = R_errs[:,3] .- R_errs[:,1]
# criteria = dif .!= 10
criteria = gaps .> 1e-5

# idx = 1574 # 1033
for idx = 1:sum(criteria)

    # main data
    gap = gaps[criteria][idx]
    R_err = R_errs[criteria,:][idx,:]
    obj = objs[criteria,:][idx,:]
    ℒ, q_ts, q_mn, q_scf, prob, gt, y, qu = datas[criteria][idx]
    q_gt = rotm2quat(gt.R)

    # matrices
    mat_scf = ℒ(q_scf)
    mat_ts = ℒ(q_ts)

    """
    Full objective value.
    """
    function ℒ2(q) # \scrL
        lam = 0.
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

    ## qu stuff
    # if length(qu) > 1
    #     printstyled("($idx) $(length(qu))\n")
    # end

    ## Observations
    # 1. Which eigenvector?
    vecs_scf = eigvecs(mat_scf)
    if abs(abs(q_scf'*vecs_scf[:,1]) - 1) < 1e-8
        # printstyled(@sprintf "(%d) SCF used v1\n" idx; color=:green)
    else
        printstyled(@sprintf "(%d) SCF used v2\n" idx; color=:red)
    end
    # 2. First order conditions met by q_ts?
    vecs_ts = eigvecs(mat_ts)
    if abs(abs(q_ts'*vecs_ts[:,1]) - 1) < 1e-8
        # printstyled(@sprintf "(%d) SDP meets first order conditions\n" idx; color=:green)
    else
        # printstyled(@sprintf "(%d) SDP violates first order conditions\n" idx; color=:blue)
    end
    # 3. Local solution of q_ts does not match q_scf
    q_local = scf(ℒ, q0=q_ts)
    if abs(abs(q_local'*q_scf) - 1) < 1e-5
        # printstyled(@sprintf "(%d) Local solution matches q_scf\n" idx; color=:green)
    else
        # printstyled(@sprintf "(%d) Local solution different from q_scf\n" idx; color=:red)
    end
    # 4. Hierarchy of objective values
    if (obj[1] - obj[3] < 0.1) && (obj[3] - ℒ2(q_local) < 0.1) && (ℒ2(q_local) - ℒ2(q_ts) < 0.1)
        # printstyled(@sprintf "(%d) %.2f < %.2f < %.2f < %.2f\n" idx obj[1] obj[3] ℒ2(q_local) ℒ2(q_ts); color=:green)
    else
        # printstyled(@sprintf "(%d) %.2f < %.2f < %.2f < %.2f\n" idx obj[1] obj[3] ℒ2(q_local) ℒ2(q_ts); color=:red)
    end
    # 5. length of qu
    if length(qu) > 3
        printstyled("($idx) Found more than 3 possibilities!\n", color=:red)
    end
    # 6. distance of qu
    if length(qu) > 1
        dists = []
        for i = 2:size(qu)[1]
            append!(dists, min(norm(qu[1] - qu[i]), norm(qu[1] - -qu[i])))
            for j = 3:size(qu)[1]
                append!(dists, min(norm(qu[2] - qu[j]), norm(qu[2] - -qu[j])))
            end
        end
        printstyled(@sprintf "(%d) %d options; min distance: %.2f\n" idx length(qu) minimum(dists); color=:blue)
    end
    # obj
    # if length(qu) > 1
    #     objs_qu = [ℒ2(quat) for quat in qu]
    #     if argmin(objs_qu) != 1
    #         printstyled(@sprintf "(%d) Max eigenvector not minimum.\n" idx; color=:red)
    #     end
    # end
end

# # find second eigenvector
# q_ans = zeros(4)
# for i = 1:1000
#     good = false
#     q_test = normalize(randn(4))
#     for j = 1:1e3
#         mat = ℒ(q_test)
#         q_new = eigvecs(mat)[:,2]
#         dir_dif = abs(abs(q_test'*q_new) - 1)
#         if dir_dif < 1e-4
#             println("SCF converged ($i guesses).")
#             good=true
#             break
#         elseif (j == 5) && (dir_dif > 0.1)
#             # give up
#             break
#         end
#         q_test = q_new
#     end
#     if good
#         global q_ans = q_test
#         break
#     end
# end
# _, e = rotm2axang(gt.R'*quat2rotm(q_ans)); e *= 180/π

## plot!
# import Plots
# Plots.quiver(zeros(3),zeros(3),zeros(3),quiver=(gt.R[1,:], gt.R[2,:], gt.R[3,:]); color=[:red; :green; :blue])
# R = quat2rotm(q_ts)
# Plots.quiver!(zeros(3),zeros(3),zeros(3),quiver=(R[1,:], R[2,:], R[3,:]); color=[:red; :green; :blue])