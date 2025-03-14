# test convergence of SCF

function setup(σm)
    # generate problem
    prob, gt, y = genproblem(σm=σm)
    lam = 0.0
    weights = ones(prob.N)

    K = prob.K
    N = prob.N
    # still need to add in weights

    ## eliminate position
    ybar = mean(eachcol(y))
    Bbar = mean(eachslice(prob.B,dims=2))

    ## eliminate shape
    Bc = eachslice(prob.B,dims=2) .- [Bbar]
    yc = reduce(hcat, eachcol(y) .- [ybar])
    Bc2 = sum([Bc[i]'*Bc[i] for i = 1:N])
    A = 2*(Bc2 + lam*I)
    invA = inv(A)
    # Schur complement
    c1 = 2*(invA - invA*ones(K)*inv(ones(K)'*invA*ones(K))*ones(K)'*invA)
    c2 = invA*ones(K)*inv(ones(K)'*invA*ones(K))

    ## Lagrangian terms
    # constant
    L = sum([yc[:,i]'*yc[:,i] for i = 1:N])
    L += c2'*Bc2*c2
    L += lam*(c2'*c2)
    # quadratic in q
    L1 = zeros(4,4)
    L2 = zeros(4,4)
    L3 = zeros(4,4)
    for i = 1:N
        L1 += -2*Ω1(yc[:,i])'*Ω2(Bc[i]*c2)
        if lam > 0.
            L3 +=  2*lam*Ω1(yc[:,i])'*Ω2(Bc[i]*c1'*c2)
            L2 +=  2*Ω1(yc[:,i])'*Ω2(Bc[i]*c1'*Bc2*c2) # (c1'*Bc2*c2 = 0 if lam=0)
        end
    end
    L1 += L1'
    L2 += L2'
    L3 += L3'
    # quartic in q
    function Lquartic(q)
        L4 = zeros(4,4)
        L5 = zeros(4,4)
        L6 = zeros(4,4)
        R = quat2rotm(q)
        Bry = sum([Bc[j]'*R'*yc[:,j] for j = 1:N])
        for i = 1:N
            if lam > 0.
                L4 += 2*lam*Ω1(yc[:,i])'*Ω2(Bc[i]*c1'*c1*Bry)
            end
            L5 += -2*Ω1(yc[:,i])'*Ω2(Bc[i]*(c1+c1')*Bry)
            L6 += 2*Ω1(yc[:,i])'*Ω2(Bc[i]*c1'*Bc2'*c1*Bry)
        end
        L4 += L4'
        L5 += L5'
        L6 += L6'
        return L4 + L5 + L6
    end

    # objective
    obj(q) = q'*(1/2*(L1 + L2 + L3) + 1/4*Lquartic(q))*q + L
    # Lagrangian
    ℒ(q) = L1 + L2 + L3 + Lquartic(q)

    return ℒ, obj, gt
end

# parameters
σm = 1.0
repeats = 100000

bad = []
bad2 = []
for i = 1:repeats
    L, obj, gt = setup(σm)

    ## solve via SCF
    function scf(ℒ, obj; eigval=1, q0=nothing)
        q_scf = q0
        q_log = [q0]
        obj_log = [obj(q0)]
        if isnothing(q0)
            q_scf = normalize(randn(4))
        end
        for i = 1:Int(1e6)
            mat = ℒ(q_scf)
            q_new = eigvecs(mat)[:,eigval]
            normalize!(q_new)

            push!(q_log, q_new)
            append!(obj_log, obj(q_new))

            if abs(abs(q_scf'*q_new) - 1) < 1e-8
                # println("SCF converged ($i iterations).")
                q_scf = q_new
                break
            end
            q_scf = q_new
        end
        return q_scf, q_log, obj_log
    end
    q_guess = normalize(randn(4))
    q_scf, q_log, obj_log = scf(L, obj; q0=q_guess)

    # check monotonic decreasing:
    qs = reduce(hcat, q_log)
    norm_seq = min.(norm.(eachcol(qs) .- [q_scf]), norm.(eachcol(qs) .+ [q_scf])) # chordal distance (analog)
    function angdif(q)
        ΔR = quat2rotm(q_scf)'*quat2rotm(q)
        _, e = rotm2axang(clamp.(ΔR, -1., 1.))
        return e
    end
    norm_seq2 = angdif.(eachcol(qs))

    if !issorted(norm_seq,rev=true)
        printstyled("($i) Norms not monotonic.\n"; color=:red)
        push!(bad, norm_seq)
        push!(bad2, norm_seq2)
    end
    if !issorted(obj_log, rev=true)
        printstyled("($i) Objectives not monotonic.\n"; color=:red)
    end
end

# import Plots
Plots.plot(obj_log, label=false, title="Objective Log")
Plots.plot!(yscale=:log10, legend=:bottom, xlabel="Iteration", ylabel="Objective")