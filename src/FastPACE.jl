module FastPACE

using LinearAlgebra
using Statistics

using SimpleRotations

# Files
include("pace_problem.jl")
export genproblem

include("pace_solvers.jl")
export solvePACE_SCF

include("pace_certifier.jl")
export certify_rotmat

end # module FastPACE
