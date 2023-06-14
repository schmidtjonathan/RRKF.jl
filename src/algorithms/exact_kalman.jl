# Kalman filter in non-square-root form


function kf_predict(μ, Σ, Φ, Q)
    μ⁻ = Φ * μ
    Σ⁻ = Φ * Σ * Φ' + Q
    return μ⁻, Σ⁻
end



function kf_predict_and_backwards_kernel(μ, Σ, Φ, Q)
    μ⁻ = Φ * μ
    Σ⁻ = Φ * Σ * Φ' + Q

    G = Σ * Φ' / Symmetric(Σ⁻)
    b = μ - G * μ⁻
    C = Σ - G * Σ⁻ * G'

    return (μ⁻, Σ⁻), (G, b, C)
end


function kf_correct(μ⁻, Σ⁻, C, R, y; compute_likelihood = false)
    cross_cov = Σ⁻ * C'
    S = Symmetric(C * cross_cov + R)
    S_chol = cholesky(S)
    measured = C * μ⁻

    residual = y - measured
    Sinv_x_residual = (S_chol \ residual)
    Δμ = cross_cov * Sinv_x_residual
    μ = μ⁻ + Δμ

    I_KC = I - cross_cov * (S_chol \ C)

    chol_R = cholesky(R)
    krkt_rsqrt = cross_cov * (S_chol \ chol_R.L)
    Σ = I_KC * Σ⁻ * I_KC' + krkt_rsqrt * krkt_rsqrt'

    loglik = if compute_likelihood
        -0.5 * (residual' * Sinv_x_residual + logdet(S_chol))
    else
        0.0
    end

    return μ, Σ, loglik
end




function kf_smooth(filter_sol::FilteringSolution{CT}) where {CT<:AbstractMatrix}
    smoothed_sol = SmoothingSolution{CT}()

    append_step!(smoothed_sol, filter_sol.t[end], filter_sol.μ[end], filter_sol.Σ[end])

    for k in reverse(2:length(filter_sol))
        G, b, C = filter_sol.backwards_transitions[k]
        if ismissing(G) || ismissing(b) || ismissing(C)
            error(
                "Found no backward transition at step $k / $(length(filter_sol)), time t = $(filter_sol.t[k])",
            )
        end

        ξ_next, Λ_next = smoothed_sol[1]
        ξ, Λ = kf_predict(ξ_next, Λ_next, G, C)
        ξ = ξ + b

        append_step!(smoothed_sol, filter_sol.t[k-1], ξ, Λ)
    end
    return smoothed_sol
end





# Square-root Kalman filter


function sqrt_kf_predict(
    m::AbstractVector,
    CL::LeftMatrixSqrt,
    A::AbstractMatrix,
    QL::LeftMatrixSqrt,
)
    mnew = A * m
    CLnew = LeftMatrixSqrt(LowerTriangular(qr([A * CL.factor QL.factor]').R'))
    return mnew, CLnew
end



function sqrt_kf_predict_and_backwards_kernel(m, CL, A, QL)
    D = length(m)
    R = qr([QL.factor A*CL.factor; zero(A') CL.factor]').R

    mnew = A * m
    CLpred = LowerTriangular(R[1:D, 1:D]')

    G = R[1:D, D+1:end]' / CLpred
    c = m - G * mnew
    ΛL = LowerTriangular(R[D+1:end, D+1:end]')

    return (mnew, LeftMatrixSqrt(CLpred)), (G, c, LeftMatrixSqrt(ΛL))
end


function sqrt_kf_correct(
    m::AbstractVector,
    CL::LeftMatrixSqrt,
    H::AbstractMatrix,
    RL::LeftMatrixSqrt,
    y::AbstractVector;
    compute_likelihood = false,
)
    d, D = size(H)

    y_hat = H * m

    R = qr([RL.factor H*CL.factor; zero(H') CL.factor]').R
    @assert istril(R[1:d, 1:d]')
    @assert istril(R[d+1:end, d+1:end]')

    SL = LowerTriangular(R[1:d, 1:d]')
    cholfac_SL = copy(SL)

    # Convert to valid Cholesky factor
    for (row_ind, sgn) in enumerate(sign.(diag(SL)))
        cholfac_SL[row_ind, :] .*= sgn
    end
    chol_SL = Cholesky(cholfac_SL)

    residual = (y - y_hat)
    Sinv_x_residual = (SL \ residual)

    mnew = m + R[1:d, d+1:end]' * Sinv_x_residual
    CLnew = LowerTriangular(R[d+1:end, d+1:end]')

    loglik = if compute_likelihood
        -0.5 * (residual' * (chol_SL \ residual) + logdet(chol_SL))
    else
        0.0
    end

    return mnew, LeftMatrixSqrt(CLnew), loglik
end





function qr_smooth(filter_sol::FilteringSolution{LeftMatrixSqrt})
    smoothed_sol = SmoothingSolution{LeftMatrixSqrt}()

    append_step!(smoothed_sol, filter_sol.t[end], filter_sol.μ[end], filter_sol.Σ[end])

    for k in reverse(2:length(filter_sol))
        G, b, sqrt_C = filter_sol.backwards_transitions[k]
        if ismissing(G) || ismissing(b) || ismissing(sqrt_C)
            error(
                "Found no backward transition at step $k / $(length(filter_sol)), time t = $(filter_sol.t[k])",
            )
        end

        ξ_next, sqrt_Λ_next = smoothed_sol[1]
        ξ, sqrt_Λ = sqrt_kf_predict(ξ_next, sqrt_Λ_next, G, sqrt_C)
        ξ = ξ + b

        append_step!(smoothed_sol, filter_sol.t[k-1], ξ, sqrt_Λ)
    end
    return smoothed_sol
end
