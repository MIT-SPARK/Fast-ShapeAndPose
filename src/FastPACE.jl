module FastPACE

using LinearAlgebra
using Statistics
using StaticArrays
using Printf

using JuMP
using MosekTools
import Manifolds
import Manopt

using SimpleRotations

# make dictionary callable
(d::Dict)(k) = d[k] 

# Files
include("problem.jl")
export genproblem

include("scf_solver.jl")
export solvePACE_SCF

include("certifier.jl")
export certify_rotmat

include("baseline_solvers.jl")
export solvePACE_SDP, solvePACE_Manopt, solvePACE_GN

include("run_synthetic.jl")
export gendata, runlocal, runlocaliter, proj_quat

include("gnc.jl")
export gnc, gnc_wrapper

end # module FastPACE
