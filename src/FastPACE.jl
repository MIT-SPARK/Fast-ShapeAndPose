module FastPACE

using LinearAlgebra
using Statistics
using StaticArrays

using SimpleRotations

# Files
include("problem.jl")
export genproblem

include("scf_solver.jl")
export solvePACE_SCF

include("certifier.jl")
export certify_rotmat

end # module FastPACE
