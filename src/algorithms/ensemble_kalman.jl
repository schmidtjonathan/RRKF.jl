ensemble_mean(ens::AbstractMatrix) = vec(mean(ens, dims = 2))

centered_ensemble(ens::AbstractMatrix) = ens .- ensemble_mean(ens)

function ensemble_cov(ens::AbstractMatrix)
    A = centered_ensemble(ens)
    N_sub_1 = size(ens, 2) - 1
    return (A * A') / N_sub_1
end

function ensemble_mean_cov(ens::AbstractMatrix)
    m = ensemble_mean(ens)
    A = ens .- m
    N_sub_1 = size(ens, 2) - 1
    C = (A * A') / N_sub_1
    return m, C
end

function ensemble_mean_sqrt_cov(ens::AbstractMatrix)
    m = ensemble_mean(ens)
    A = ens .- m
    sqrt_N_sub_1_rec = 1.0 ./ sqrt(size(ens, 2) - 1)
    return m, LeftMatrixSqrt(sqrt_N_sub_1_rec * A)
end

function _calc_PH_HPH(ensemble, H)
    D, N = size(ensemble)
    d = size(H, 1)
    A = centered_ensemble(ensemble)
    PH = zeros(D, d)
    HPH = zeros(d, d)
    @inbounds @simd for i = 1:N
        x_i = A[:, i]
        meas = H * x_i
        PH .+= x_i * meas'
        HPH .+= meas * meas'
    end
    return PH / (N - 1.0), HPH / (N - 1.0)
end


function enkf_predict(ensemble, Φ, Q_sqrt::LeftMatrixSqrt)
    D, N = size(ensemble)
    sampled_proc_noise = Q_sqrt.factor * rand(MvNormal(I(D)), N)
    forecast_ensemble = Φ * ensemble + sampled_proc_noise

    return forecast_ensemble
end


function enkf_correct(
    forecast_ensemble,
    H,
    measurement_noise_dist::MvNormal,
    y;
    compute_likelihood = false,
)
    N = size(forecast_ensemble, 2)
    HX = H * forecast_ensemble
    data_plus_noise = rand(measurement_noise_dist, N) .+ y
    residual = data_plus_noise - HX
    PH, HPH = _calc_PH_HPH(forecast_ensemble, H)

    Ŝ = HPH + measurement_noise_dist.Σ
    S_chol = cholesky(Symmetric(Ŝ))
    ensemble = forecast_ensemble + PH * (S_chol \ residual)

    loglik = if compute_likelihood
        innovation_vector = y - ensemble_mean(HX)
        -0.5 * (innovation_vector' * (S_chol \ innovation_vector) + logdet(S_chol))
    else
        0.0
    end

    return ensemble, loglik
end


function etkf_correct(
    forecast_ensemble,
    H,
    measurement_noise_dist::MvNormal,
    y;
    compute_likelihood = false,
)
    !compute_likelihood || error("ETKF likelihood is not implemented.")

    N = size(forecast_ensemble, 2)
    R_chol = cholesky(measurement_noise_dist.Σ)
    H̃ = R_chol.L \ H

    forecast_mean = ensemble_mean(forecast_ensemble)
    Z_f = forecast_ensemble .- forecast_mean
    HA = H̃ * Z_f
    HX = H̃ * forecast_ensemble

    Zy = (1.0 / sqrt(N - 1)) * HA
    C, G, F = svd(Zy', full = size(Zy, 1) < N, alg = LinearAlgebra.QRIteration())
    G[G.<eps(eltype(forecast_ensemble))] .= eps(eltype(forecast_ensemble))
    if length(G) < N
        G_I = vcat(G, zeros(N - length(G)))
        G_I = Diagonal(1.0 ./ sqrt.(1.0 .+ G_I .^ 2))
    else
        G_I = Diagonal(1.0 ./ sqrt.(1.0 .+ G .^ 2))
    end
    T = C * G_I # * C'   # The * C' comes from Eq. (13) in Sakov, Oke (2008) "Implications..."

    Z_a = (1.0 / sqrt(N - 1)) * Z_f * T

    # http://pdaf.awi.de/files/SEIK-ETKF-ESTKF.pdf -> Eq. (7)
    # where their A = T * T'
    # Also, see e.g. Eq. (27) in https://journals.ametsoc.org/view/journals/mwre/147/8/mwr-d-18-0210.1.xml
    K = Z_a * (H̃ * Z_a)'

    whitened_residual = (R_chol.L \ y - ensemble_mean(HX))
    analysis_mean = forecast_mean .+ K * whitened_residual
    analysis_ensemble = analysis_mean .+ sqrt(N - 1) * Z_a

    return analysis_ensemble, 0.0
end
