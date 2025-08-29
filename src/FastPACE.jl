module FastPACE

using LinearAlgebra
using Statistics
using StaticArrays

using JuMP
using MosekTools

using SimpleRotations

# Files
include("problem.jl")
export genproblem

include("scf_solver.jl")
export solvePACE_SCF

include("certifier.jl")
export certify_rotmat

include("baseline_solvers.jl")
export solvePACE_SDP

end # module FastPACE
