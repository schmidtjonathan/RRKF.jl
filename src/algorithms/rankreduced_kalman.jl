function tsvd(A, r)
    U, σs, V = try
        svd(A)
    catch e
        e
        if isa(e, LAPACKException)
            svd(A; alg = LinearAlgebra.QRIteration())
        else
            rethrow(e)
        end
    end

    U_partial = U[:, 1:r]
    σs_partial = σs[1:r]
    V_partial = V[:, 1:r]
    return U_partial, σs_partial, V_partial
end

export tsvd




function rrkf_predict(μ, Σ_sqrt::LeftMatrixSqrt, Φ, Q_sqrt::LeftMatrixSqrt, r)
    next_mean = Φ * μ
    U_Π, s_Π, V_Π = tsvd([Φ * Σ_sqrt.factor Q_sqrt.factor], r)
    next_cov_sqrt = LeftMatrixSqrt(U_Π * Diagonal(s_Π))
    return next_mean, next_cov_sqrt
end


function rrkf_backwards_prediction(ξ_next, sqrt_Λ_next, G, b, sqrt_P, r)
    Σ_sqrt_factor, Γ, Π_inv_sqrt_factor = G

    ξ = Σ_sqrt_factor * (Γ * (Π_inv_sqrt_factor' * ξ_next)) + b

    G_times_sqrt_Λ_next = Σ_sqrt_factor * (Γ * (Π_inv_sqrt_factor' * sqrt_Λ_next.factor))
    U_Λ, s_Λ, V_Λ = tsvd([G_times_sqrt_Λ_next sqrt_P.factor], r)
    D_Λ = Diagonal(s_Λ)
    Λ_sqrt = U_Λ * D_Λ

    return ξ, LeftMatrixSqrt(Λ_sqrt)
end


function rrkf_predict_and_backwards_kernel(
    μ,
    Σ_sqrt::LeftMatrixSqrt,
    Φ,
    Q_sqrt::LeftMatrixSqrt,
    r,
)
    next_mean = Φ * μ
    U_Π, s_Π, V_Π = tsvd([Φ * Σ_sqrt.factor Q_sqrt.factor], r)
    D_Π = Diagonal(s_Π)
    next_cov_sqrt = LeftMatrixSqrt(U_Π * D_Π)


    # BACKWARDS KERNEL
    tolerance = size(D_Π, 1) * eps(eltype(D_Π))
    Π_inveigvals = pinv(D_Π, tolerance)
    Π_inv_sqrt_factor = U_Π * Π_inveigvals

    Σ_sqrt_factor = Σ_sqrt.factor

    Γ = Σ_sqrt_factor' * Φ' * Π_inv_sqrt_factor

    b = μ - Σ_sqrt_factor * (Γ * (Π_inv_sqrt_factor' * next_mean))

    I_minus_GΦ_Σsqrt =
        Σ_sqrt_factor - Σ_sqrt_factor * (Γ * (Π_inv_sqrt_factor' * (Φ * Σ_sqrt_factor)))
    GQ_sqrt = Σ_sqrt_factor * (Γ * (Π_inv_sqrt_factor' * Q_sqrt.factor))
    U_P, s_P, V_P = tsvd([I_minus_GΦ_Σsqrt GQ_sqrt], r)
    P_sqrt = U_P * Diagonal(s_P)

    G = (Σ_sqrt_factor, Γ, Π_inv_sqrt_factor)

    return (next_mean, next_cov_sqrt), (G, b, LeftMatrixSqrt(P_sqrt))
end



function rrkf_correct(
    μ⁻,
    Π_sqrt::LeftMatrixSqrt,
    C,
    R_sqrt::LeftMatrixSqrt,
    y,
    r;
    compute_likelihood = false,
)
    if r <= size(C, 1)
        _rrkf_correct_rsmallerm(
            μ⁻,
            Π_sqrt,
            C,
            R_sqrt,
            y;
            compute_likelihood = compute_likelihood,
        )
    else
        _rrkf_correct_rlargerm(
            μ⁻,
            Π_sqrt,
            C,
            R_sqrt,
            y;
            compute_likelihood = compute_likelihood,
        )
    end
end



function _rrkf_correct_rsmallerm(
    μ⁻,
    Π_sqrt::LeftMatrixSqrt,
    C,
    R_sqrt::LeftMatrixSqrt,
    y;
    compute_likelihood,
)

    y_hat = C * μ⁻

    F_S = svd((R_sqrt.factor \ (C * Π_sqrt.factor))')
    D_S = Diagonal(F_S.S)

    K_ = Π_sqrt.factor * ((F_S.U / (I + D_S^2)) * D_S)

    ϵ = R_sqrt.factor \ (y - y_hat)

    μ = μ⁻ + K_ * (F_S.Vt * ϵ)

    Σ_sqrt = Π_sqrt.factor * F_S.U / Diagonal(sqrt.(1 .+ diag(D_S) .^ 2))

    loglik = if compute_likelihood
        trm = ϵ' * F_S.V * D_S
        -logdet(R_sqrt.factor) - 0.5 * sum(log.(diag(D_S) .^ 2 .+ 1.0)) - 0.5 * ϵ' * ϵ +
        0.5 * (trm * inv(D_S^2 + I) * trm')
    else
        0.0
    end
    return μ, LeftMatrixSqrt(Σ_sqrt), loglik
end



function _rrkf_correct_rlargerm(
    μ⁻,
    Π_sqrt::LeftMatrixSqrt,
    C,
    R_sqrt::LeftMatrixSqrt,
    y;
    compute_likelihood,
)

    statedim, r = size(Π_sqrt.factor)

    y_hat = C * μ⁻

    F_S = svd([C * Π_sqrt.factor R_sqrt.factor])
    D_S = Diagonal(F_S.S)

    K̃ = (C * Π_sqrt.factor)' * F_S.U / D_S
    K = Π_sqrt.factor * K̃ / D_S * F_S.U'


    ϵ = (y - y_hat)
    μ = μ⁻ + K * ϵ

    F_K = svd(K̃, full = true)

    D_K_full = vcat(F_K.S, zeros(r - length(F_K.S)))
    if any(D_K_full .>= 1.0)
        @warn "Weird shit happened."
    end
    D_K_full_1_sqrt = Diagonal(sqrt.(1.0 .- D_K_full .^ 2))

    Σ_sqrt = Π_sqrt.factor * F_K.U * D_K_full_1_sqrt

    loglik = if compute_likelihood
        tolerance = size(D_S, 1) * eps(eltype(D_S))
        S_inveigvals = pinv(D_S^2, tolerance)
        half_logdetS = sum(log.(diag(D_S)))
        -0.5 * (ϵ' * F_S.U * (S_inveigvals * (F_S.U' * ϵ))) - half_logdetS
    else
        0.0
    end

    return μ, LeftMatrixSqrt(Σ_sqrt), loglik
end


function rrkf_smooth(filter_sol::FilteringSolution{LeftMatrixSqrt})
    smoothed_sol = SmoothingSolution{LeftMatrixSqrt}()

    append_step!(smoothed_sol, filter_sol.t[end], filter_sol.μ[end], filter_sol.Σ[end])
    r = size(filter_sol.Σ[end].factor, 2)

    for k in reverse(2:length(filter_sol))
        G, b, sqrt_C = filter_sol.backwards_transitions[k]
        if ismissing(G) || ismissing(b) || ismissing(sqrt_C)
            error(
                "Found no backward transition at step $k / $(length(filter_sol)), time t = $(filter_sol.t[k])",
            )
        end

        ξ_next, sqrt_Λ_next = smoothed_sol[1]
        ξ, sqrt_Λ = rrkf_backwards_prediction(ξ_next, sqrt_Λ_next, G, b, sqrt_C, r)

        append_step!(smoothed_sol, filter_sol.t[k-1], ξ, sqrt_Λ)
    end
    return smoothed_sol
end
