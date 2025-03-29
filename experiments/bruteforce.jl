## Brute force a solution using only GNC for outlier rejection
# run simulation first I guess
# TODO: move to function


# define problem
prob, gt, y = genproblem(σm=σm)

# add bad associations
B_new = repeat(prob.B, 1, prob.N, 1)
y_new = repeat(y, inner=[1,prob.N])

lam = 0.0
weights = ones(prob.N)