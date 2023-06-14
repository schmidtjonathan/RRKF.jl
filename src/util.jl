function symmetrize_matrix(A)
    return 0.5 * (A + A')
end

function rand_orth_mat(n, m)
    A = randn(n, m)
    F = qr!(A)
    return Matrix(F.Q)
end


vecvec2mat(vs) = reduce(hcat, vs)'
intersperse(vs) = reduce(hcat, vs)'[:]
tuple2vec(t) = [_t for _t in t]

export vecvec2mat
export intersperse
export tuple2vec


num_zeros(A) = sum(iszero.(A))
function sparse_criterion(A; frac_zeros = 2 / 3)
    numelA = size(A) |> prod
    return num_zeros(A) >= frac_zeros * numelA
end
