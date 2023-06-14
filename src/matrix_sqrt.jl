import Base: Matrix, size, copy
import LinearAlgebra: diag

struct LeftMatrixSqrt
    factor::AbstractMatrix
end

function LeftMatrixSqrt(factor::LinearAlgebra.UpperTriangular)
    return LeftMatrixSqrt(factor')
end

function LeftMatrixSqrt(factor::LinearAlgebra.Cholesky)
    return LeftMatrixSqrt(factor.U')
end

function copy(rms::LeftMatrixSqrt)
    return LeftMatrixSqrt(copy(rms.factor))
end

function Base.Matrix(rms::LeftMatrixSqrt)
    return rms.factor * rms.factor'
end

function LinearAlgebra.diag(rms::LeftMatrixSqrt)
    n, r = size(rms.factor)
    return [sum(rms.factor[i, :] .^ 2) for i = 1:n]
end

function Base.size(rms::LeftMatrixSqrt)
    return size(rms.factor)
end


function Base.size(rms::LeftMatrixSqrt, i)
    return size(rms.factor, i)
end

export LeftMatrixSqrt
