## Certify global optimality of solutions to shape & pose estimation
# Lorenzo Shaikewitz, 6/27/2025

"""
Certify based on the rotation matrix representation.

No / minimal performance improvements
"""
function certify_rotmat(prob::Problem, soln::Solution, y, weights, lam=0.; tol=1e-3)
    K = prob.K
    N = prob.N

    if sum(weights .!= 0) <= prob.K && lam == 0
        lam = 0.1
        @warn "setting lam = 0.1 since N ≤ K."
    end

    ## eliminate position
    ȳ = sum(weights .* eachcol(y)) ./ sum(weights)
    B̄ = sum(weights .* eachslice(prob.B, dims=2)) ./ sum(weights)

    ## eliminate shape
    B̄s = (eachslice(prob.B, dims=2) .- [B̄]).*sqrt.(weights)
    ȳs = reduce(hcat, (eachcol(y) .- [ȳ]).*sqrt.(weights))
    B̂² = sum([B̄s[i]'*B̄s[i] for i = 1:N])
    H = 2(B̂² + lam*I)
    invH = inv(H)
    # Schur complement
    C1 = 2(invH - invH*ones(K)*inv(ones(K)'*invH*ones(K))*ones(K)'*invH)
    c2 = invH*ones(K)*inv(ones(K)'*invH*ones(K))

    ## Objective (rotation matrix representation)
    P = zeros(9,9) # permutation matrix vec(R') => vec(R)
    P[1,1] = P[5,5] = P[9,9] = 1
    P[2,4] = P[3,7] = P[4,2] = P[6,8] = P[7,3] = P[8,6] = 1
    P = P' # vec(R) => vec(R')

    # (A*(vec(R) - b))'*(A*(vec(R) - b))
    A = zeros(9,9)
    b = zeros(9)

    kronsum = sum([kron(ȳs[:,j]',B̄s[j]')*P for j = 1:N])

    # lam regularized terms
    b += (lam*c2'*C1*kronsum)'
    A += lam*(C1*kronsum)'*(C1*kronsum)

    # main objective
    A += (C1*kronsum)'*B̂²*(C1*kronsum)
    b += (c2'*B̂²*C1*kronsum)'
    A += -2*kronsum'*C1*kronsum
    b += -(c2'*kronsum)'
    
    # VERIFIED
    C = [A  b;  b'  0] # objective matrix [vec(R); 1]'*C*[vec(R); 1]

    xlocal = [vec(soln.R); 1]
    As = constraintsO3()
    A1 = zeros(10,10); A1[10,10] = 1
    push!(As, A1)

    # only use O(3) constraints to avoid violating LICQ
    # As = filter!(!isempty, As)
    A = reduce(hcat, As .* [xlocal])
    # lagrange multipliers
    λ = A \ (C*xlocal)
    # certificate matrix
    S = C - sum(λ .* As)

    return eigvals(S)[1] > -tol
end


## Quadratic forms of O(3) constraints
# right hand rule constraints violate LICQ!
const O3_1 = SMatrix{10,10}(vcat(hcat(diagm(ones(3)), zeros(3,7)), zeros(6,10), hcat(zeros(1,9), -1))...)
const O3_2 = SMatrix{10,10}(vcat(zeros(3,10), hcat(zeros(3,3), diagm(ones(3)), zeros(3,4)), zeros(3,10), hcat(zeros(1,9), -1))...)
const O3_3 = SMatrix{10,10}(vcat(zeros(6,10), hcat(zeros(3,6), diagm(ones(3)), zeros(3,1)), hcat(zeros(1,9), -1))...)
const O3_4 = SMatrix{10,10}(vcat(hcat(zeros(3,3), 0.5*diagm(ones(3)), zeros(3,4)),
                                 hcat(0.5*diagm(ones(3)), zeros(3,7)),
                                 zeros(4,10))...)
const O3_5 = SMatrix{10,10}(vcat(hcat(zeros(3,6), 0.5*diagm(ones(3)), zeros(3,1)),
                                 zeros(3,10),
                                 hcat(0.5*diagm(ones(3)), zeros(3,7)),
                                 zeros(1,10))...)
const O3_6 = SMatrix{10,10}(vcat(zeros(3,10),
                                 hcat(zeros(3,6), 0.5*diagm(ones(3)), zeros(3,1)),
                                 hcat(zeros(3,3), 0.5*diagm(ones(3)), zeros(3,4)),
                                 zeros(1,10))...)
const AONE = @SMatrix [(i,j)==(10,10) ? 1 : 0 for i in 1:10, j in 1:10]
const O3_CONSTRAINTS = [O3_1, O3_2, O3_3, O3_4, O3_5, O3_6, AONE]



"""
Quadratic forms of O(3) constraints
"""
function constraintsO3()
    q_eqs = []
    # r'*r = 1
    for i = 0:2
        H = zeros(9,9)
        H[3*i+1,3*i+1] = 1.; H[3*i+2,3*i+2] = 1.; H[3*i+3,3*i+3] = 1.
        b = zeros(9)
        s = -1
        Q = [H  b; b' s]
        push!(q_eqs, Q)
    end
    # r1'*r2 = r1'*r3 = r2'*r3 = 0
    begin
        # r1'*r2
        H = zeros(9,9)
        H[1,4] = 0.5; H[2,5] = 0.5; H[3,6] = 0.5
        H += H'
        b = zeros(9); s = 0
        Q = [H  b; b' s]
        push!(q_eqs, Q)
        # r1'*r3
        H = zeros(9,9)
        H[1,7] = 0.5; H[2,8] = 0.5; H[3,9] = 0.5
        H += H'
        b = zeros(9); s = 0
        Q = [H  b; b' s]
        push!(q_eqs, Q)
        # r2'*r3
        H = zeros(9,9)
        H[4,7] = 0.5; H[5,8] = 0.5; H[6,9] = 0.5
        H += H'
        b = zeros(9); s = 0
        Q = [H  b; b' s]
        push!(q_eqs, Q)
    end
    # cross products
    # begin
    #     # r1 x r2 - r3
    #     H = zeros(9,9)
    #     H[2,6] = 1; H[3,5] = -1; H += H'
    #     c = zeros(9); c[7] = -1.
    #     Q = [H  c; c' 0.]
    #     push!(q_eqs, Q)
    #     H = zeros(9,9)
    #     H[3,4] = 1; H[1,6] = -1; H += H'
    #     c = zeros(9); c[8] = -1.
    #     Q = [H  c; c' 0.]
    #     push!(q_eqs, Q)
    #     H = zeros(9,9)
    #     H[1,5] = 1; H[2,4] = -1; H += H'
    #     c = zeros(9); c[9] = -1.
    #     Q = [H  c; c' 0.]
    #     push!(q_eqs, Q)

    #     # r2 x r3 - r1
    #     H = zeros(9,9)
    #     H[5,9] = 1; H[6,8] = -1; H += H'
    #     c = zeros(9); c[1] = -1.
    #     Q = [H  c; c' 0.]
    #     push!(q_eqs, Q)
    #     H = zeros(9,9)
    #     H[6,7] = 1; H[4,9] = -1; H += H'
    #     c = zeros(9); c[2] = -1.
    #     Q = [H  c; c' 0.]
    #     push!(q_eqs, Q)
    #     H = zeros(9,9)
    #     H[4,8] = 1; H[5,7] = -1; H += H'
    #     c = zeros(9); c[3] = -1.
    #     Q = [H  c; c' 0.]
    #     push!(q_eqs, Q)

    #     # r3 x r1 - r2
    #     H = zeros(9,9)
    #     H[8,3] = 1; H[9,2] = -1; H += H'
    #     c = zeros(9); c[4] = -1.
    #     Q = [H  c; c' 0.]
    #     push!(q_eqs, Q)
    #     H = zeros(9,9)
    #     H[9,1] = 1; H[7,3] = -1; H += H'
    #     c = zeros(9); c[5] = -1.
    #     Q = [H  c; c' 0.]
    #     push!(q_eqs, Q)
    #     H = zeros(9,9)
    #     H[7,2] = 1; H[8,1] = -1; H += H'
    #     c = zeros(9); c[6] = -1.
    #     Q = [H  c; c' 0.]
    #     push!(q_eqs, Q)
    # end
    return q_eqs
end