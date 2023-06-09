module RRKF

using LinearAlgebra
using Random
using Kronecker
using Distributions
using ProgressMeter
using SparseArrays

import MatrixEquations: lyapc


include("util.jl")
include("matrix_sqrt.jl")
include("ssm.jl")
include("spatiotemporal_process.jl")

include("dlra.jl")

include("filtering_solution.jl")

include("algorithms/exact_kalman.jl")
include("algorithms/ensemble_kalman.jl")
include("algorithms/rankreduced_kalman.jl")

include("estimate_states.jl")

end