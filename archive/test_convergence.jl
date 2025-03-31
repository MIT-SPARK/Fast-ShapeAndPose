# test convergence of SCF

function setup(σm)
    # generate problem
    prob, gt, y = genproblem(σm=σm)
    lam = 0.0
    weights = ones(prob.N)

    K = prob.K
    N = prob.N

    ## eliminate position
    ybar = sum(weights .* eachcol(y)) ./ sum(weights)
    Bbar = sum(weights .* eachslice(prob.B,dims=2)) ./ sum(weights)

    ## eliminate shape
    Bc = (eachslice(prob.B,dims=2) .- [Bbar]).*sqrt.(weights)
    yc = reduce(hcat, (eachcol(y) .- [ybar]).*sqrt.(weights))
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
# Random.seed!(7) # produces 3 near-identical solutions
# other seeds: 14, 20, 24, 32, 41, 56, 60, 63, 78, 84
Random.seed!(14)

ℒ, obj, gt = setup(σm)

## solve via SCF
function scf(ℒ; eigval=1, q0=nothing, quat_log=nothing)
    q_scf = q0
    new = true
    if isnothing(q0)
        q_scf = normalize(randn(4))
    end
    q_log = [q_scf]
    for _ = 1:Int(1e3)
        mat = ℒ(q_scf)
        q_new = eigvecs(mat)[:,eigval]
        normalize!(q_new)

        push!(q_log, q_new)
        if !isnothing(quat_log)
            for quat in quat_log
                if abs(abs(q_new'*quat[end]) - 1) < 1e-5
                    new = false
                    break
                end
            end
            if !new
                break
            end
        end

        if abs(abs(q_scf'*q_new) - 1) < 1e-8
            q_scf = q_new
            break
        end
        q_scf = q_new
    end
    return q_scf, q_log, new
end
q_guess = normalize(randn(4))
q_scf, q_log = scf(ℒ; q0=q_guess)

# check for multiple solutions
q_logs = [q_log]
all_q_logs = [q_log]
for _ = 1:1000
    local q_scf, q_log, new = scf(ℒ; quat_log=q_logs)
    if new
        push!(q_logs, q_log)
    end
    push!(all_q_logs, q_log)
end
println("Found $(length(q_logs)) solutions.")

objs = obj.([ql[end] for ql in q_logs])

# projecting quaternions to 3D:
function proj_quat(q)
    q_proj = q[2:end]
    if q[1] < 0
        # q_proj = clamp.(1. ./ q_proj,-2,2)
        q_proj = -q_proj
    end
    return q_proj
end

# plot
import Plots
q_scf = q_logs[1][end]
# p1 = Plots.plot()
p2 = Plots.plot3d()
for q_log in all_q_logs
    local qs = reduce(hcat, q_log)
    local norm_seq = min.(norm.(eachcol(qs) .- [q_scf]), norm.(eachcol(qs) .+ [q_scf]))
    local qs = reduce(hcat, proj_quat.(q_log))

    if abs(abs(q_log[end]'*q_scf) - 1) < 1e-5
        # Plots.plot!(p1, norm_seq, label=false, lc=:blue)
        # local colors = Plots.cgrad(:blues, length(q_log), categorical = true)
        Plots.scatter3d!(p2, qs[1,:], qs[2,:], qs[3,:], label="toq1", aspect_ratio=:equal, mc=:blue) #color=collect(colors))
    else
        # Plots.plot!(p1, norm_seq, label=false, lc=:red)
        # local colors = Plots.cgrad(:heat, length(q_log), categorical = true)
        Plots.scatter3d!(p2, qs[1,:], qs[2,:], qs[3,:], label="toq2", aspect_ratio=:equal, mc=:red) #color=collect(colors))
    end
end

q1 = proj_quat(q_logs[1][end])
q1m = proj_quat(-q_logs[1][end])
Plots.scatter3d!(p2, [q1[1]], [q1[2]], [q1[3]], markershape=:cross, ms=10, label="q1")
Plots.scatter3d!(p2, [q1m[1]], [q1m[2]], [q1m[3]], markershape=:cross, ms=10, label="q1")
if length(q_logs) > 1
q2 = proj_quat(q_logs[2][end])
q2m = proj_quat(-q_logs[2][end])
Plots.scatter3d!(p2, [q2[1]], [q2[2]], [q2[3]], markershape=:cross, ms=10, label="q2")
Plots.scatter3d!(p2, [q2m[1]], [q2m[2]], [q2m[3]], markershape=:cross, ms=10, label="q2")
end
Plots.scatter3d!(p2,aspect_ratio=1,xlim=[-2,2],ylim=[-2,2],zlim=[-2,2])



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
    printstyled("Norms not monotonic.\n"; color=:red)
end
if !issorted(obj.(q_log), rev=true)
    printstyled("Objectives not monotonic.\n"; color=:red)
end


## Alg for sampling away from points
grid = randn(4,1000)
grid = normalize.(eachcol(grid))
dists = 100*ones(1000)

function furthest()
    # sample point furthest from all points visited
    global grid, visited, dists
    # 1) sample
    sample = grid[argmax(dists)]
    # 2) update distances
    dists = min.(dists, norm.(grid .- [sample]))
    return sample
end

q_guess = normalize(randn(4))
q_scf, q_log = scf(ℒ; q0=q_guess)

# check for multiple solutions
q_logs = [q_log]
all_q_logs = [q_log]
for iter = 1:1000
    q0_new = furthest()

    local q_scf, q_log, new = scf(ℒ; q0=q0_new, quat_log=q_logs)
    push!(all_q_logs, q_log)
    if new
        println("New min found at iteration $iter.")
        push!(q_logs, q_log)
    end
end

p3 = Plots.plot3d()
for q_log in all_q_logs
    local qs = reduce(hcat, q_log)
    local norm_seq = min.(norm.(eachcol(qs) .- [q_scf]), norm.(eachcol(qs) .+ [q_scf]))
    local qs = reduce(hcat, proj_quat.(q_log))

    if abs(abs(q_log[end]'*q_scf) - 1) < 1e-5
        # Plots.plot!(p1, norm_seq, label=false, lc=:blue)
        # local colors = Plots.cgrad(:blues, length(q_log), categorical = true)
        Plots.scatter3d!(p3, qs[1,:], qs[2,:], qs[3,:], label="toq1", aspect_ratio=:equal, mc=:blue,ms=2) #color=collect(colors))
    else
        # Plots.plot!(p1, norm_seq, label=false, lc=:red)
        # local colors = Plots.cgrad(:heat, length(q_log), categorical = true)
        Plots.scatter3d!(p3, qs[1,:], qs[2,:], qs[3,:], label="toq2", aspect_ratio=:equal, mc=:red,ms=2) #color=collect(colors))
    end
end
Plots.scatter3d!(p3, [q1[1]], [q1[2]], [q1[3]], markershape=:cross, ms=10, label="q1")
Plots.scatter3d!(p3, [q1m[1]], [q1m[2]], [q1m[3]], markershape=:cross, ms=10, label="q1")
if length(q_logs) > 1
Plots.scatter3d!(p3, [q2[1]], [q2[2]], [q2[3]], markershape=:cross, ms=10, label="q2")
Plots.scatter3d!(p3, [q2m[1]], [q2m[2]], [q2m[3]], markershape=:cross, ms=10, label="q2")
end
Plots.scatter3d!(p3,aspect_ratio=1,xlim=[-2,2],ylim=[-2,2],zlim=[-2,2])