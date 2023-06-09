abstract type AbstractFilteringAlgorithm end

struct KalmanFilter <: AbstractFilteringAlgorithm end

struct SqrtKalmanFilter <: AbstractFilteringAlgorithm end

struct RankReducedKalmanFilter <: AbstractFilteringAlgorithm
    nvals::Int64
    DLR_steps::Int64
end

struct EnsembleKalmanFilter <: AbstractFilteringAlgorithm
    ensemble_size::Int64
    correct_function
end


function _measmod_without_nans(ssm::StateSpaceModel, y::AbstractVector)
    cur_not_nan_idcs = .~isnan.(y)
    cur_H = ssm.H[cur_not_nan_idcs, :]
    Hₜ = if (issparse(ssm.H) || sparse_criterion(cur_H))
        sparse(cur_H)
    else
        cur_H
    end
    yₜ = y[cur_not_nan_idcs]
    Rₜ = ssm.R[cur_not_nan_idcs, cur_not_nan_idcs]
    return Hₜ, Rₜ, yₜ
end


function estimate_states(::KalmanFilter, ssm, times, observations; smooth=false, compute_likelihood, save_all_steps=true, show_progress=true)

    if smooth && !save_all_steps
        error("Cannot smooth without saving all steps")
    end

    previous_t = if length(times) == length(observations)
        times[1]
    else
        error("Lengths of times and observations don't match")
    end

    # Initialize
    # 1. compute stationary moments
    m_stat, P_stat = stationary_moments(ssm.dynamics)
    # 2. condition on first measurement
    μ₀, Σ₀, ll₀ = if !all(isnan.(observations[1]))
        Hₜ, Rₜ, yₜ = _measmod_without_nans(ssm, observations[1])
        kf_correct(m_stat, symmetrize_matrix(P_stat), Hₜ, Rₜ, yₜ; compute_likelihood=compute_likelihood)
    else
        copy(m_stat), symmetrize_matrix(P_stat), 0.0
    end


    current_m, current_P = μ₀, Σ₀
    filter_sol = if save_all_steps
        append_step!(FilteringSolution(current_P), previous_t, current_m, current_P)
    else
        nothing
    end

    loglik = ll₀

    progbar = Progress(length(times) - 1; showspeed=true, enabled=show_progress)
    for (t, y) in zip(times[2:end], observations[2:end])
        current_dt = t - previous_t
        previous_t = t

        # Discretize
        A, Q = discretize(ssm.dynamics, current_dt)

        # Predict
        (μ⁻, Π), (G, b, backwards_S) = if smooth
            kf_predict_and_backwards_kernel(current_m, current_P, A, Q)
        else
            kf_predict(current_m, current_P, A, Q), (missing, missing, missing)
        end

        # Correct
        current_m, current_P, ll = if !all(isnan.(y))
            Hₜ, Rₜ, yₜ = _measmod_without_nans(ssm, y)
            kf_correct(μ⁻, Π, Hₜ, Rₜ, yₜ; compute_likelihood=compute_likelihood)
        else
            μ⁻, Π, 0.0
        end

        if save_all_steps
            append_step!(filter_sol, t, current_m, current_P, G, b, backwards_S)
        end
        loglik += ll

        ProgressMeter.next!(progbar)
    end

    mean_loglik = loglik / length(observations)

    # Smoothing
    smoother_sol = if smooth
        kf_smooth(filter_sol)
    else
        nothing
    end

    estimate = Dict(
        :filter => save_all_steps ? filter_sol : (t=previous_t, μ=current_m, Σ=current_P),
        :smoother => smoother_sol,
        :loglikelihood => mean_loglik
    )
    return estimate
end



function estimate_states(::SqrtKalmanFilter, ssm, times, observations; smooth=false, compute_likelihood, save_all_steps=true, show_progress=true)

    if smooth && !save_all_steps
        error("Cannot smooth without saving all steps")
    end

    previous_t = if length(times) == length(observations)
        times[1]
    else
        error("Lengths of times and observations don't match")
    end

    # Initialize
    # 1. compute stationary moments
    m_stat, P_stat = stationary_moments(ssm.dynamics)
    # 2. condition on first measurement
    μ₀, Σ₀_sqrt, ll₀ = if !all(isnan.(observations[1]))
        Hₜ, Rₜ, yₜ = _measmod_without_nans(ssm, observations[1])
        P_stat_sqrt = LeftMatrixSqrt(cholesky(Symmetric(symmetrize_matrix(P_stat))))
        R_sqrt = LeftMatrixSqrt(cholesky(Rₜ))
        μ₀, Σ₀_sqrt, ll₀ = sqrt_kf_correct(m_stat, P_stat_sqrt, Hₜ, R_sqrt, yₜ; compute_likelihood=compute_likelihood)
    else
        P_stat_sqrt = LeftMatrixSqrt(cholesky(Symmetric(symmetrize_matrix(P_stat))))
        copy(m_stat), P_stat_sqrt, 0.0
    end



    current_m, current_P_sqrt = μ₀, Σ₀_sqrt
    filter_sol = if save_all_steps
        append_step!(FilteringSolution(current_P_sqrt), previous_t, current_m, current_P_sqrt)
    else
        nothing
    end


    loglik = ll₀

    progbar = Progress(length(times) - 1; showspeed=true, enabled=show_progress)
    for (t, y) in zip(times[2:end], observations[2:end])
        current_dt = t - previous_t
        previous_t = t

        # Discretize
        A, Q_sqrt = discretize_sqrt(ssm.dynamics, current_dt)

        # Predict
        (μ⁻, Π_sqrt), (G, b, backwards_S_sqrt) = if smooth
            sqrt_kf_predict_and_backwards_kernel(current_m, current_P_sqrt, A, Q_sqrt)
        else
            sqrt_kf_predict(current_m, current_P_sqrt, A, Q_sqrt), (missing, missing, missing)
        end

        # Correct
        current_m, current_P_sqrt, ll = if !all(isnan.(y))
            Hₜ, Rₜ, yₜ = _measmod_without_nans(ssm, y)
            R_sqrt = LeftMatrixSqrt(cholesky(Rₜ))
            sqrt_kf_correct(μ⁻, Π_sqrt, Hₜ, R_sqrt, yₜ; compute_likelihood=compute_likelihood)
        else
            μ⁻, Π_sqrt, 0.0
        end

        if save_all_steps
            append_step!(filter_sol, t, current_m, current_P_sqrt, G, b, backwards_S_sqrt)
        end
        loglik += ll

        ProgressMeter.next!(progbar)
    end

    mean_loglik = loglik / length(observations)

    # Smoothing
    smoother_sol = if smooth
        qr_smooth(filter_sol)
    else
        nothing
    end

    estimate = Dict(
        :filter => save_all_steps ? filter_sol : (t=previous_t, μ=current_m, Σ=current_P_sqrt),
        :smoother => smoother_sol,
        :loglikelihood => mean_loglik
    )
    return estimate
end



function estimate_states(alg::RankReducedKalmanFilter, ssm, times, observations; smooth=false, compute_likelihood, save_all_steps=true, show_progress=true)

    if smooth && !save_all_steps
        error("Cannot smooth without saving all steps")
    end

    previous_t = if length(times) == length(observations)
        times[1]
    else
        error("Lengths of times and observations don't match")
    end

    # Initialize
    # 1. compute stationary moments
    m_stat, P_stat = stationary_moments(ssm.dynamics)

    # 2. condition on first measurement
    μ₀, Σ₀_sqrt, ll₀ = if !all(isnan.(observations[1]))
        Hₜ, Rₜ, yₜ = _measmod_without_nans(ssm, observations[1])
        U_Pstat, s_Pstat, V_Pstat = tsvd(symmetrize_matrix(P_stat), alg.nvals)

        P_stat_sqrt = LeftMatrixSqrt(U_Pstat * Diagonal(sqrt.(s_Pstat)))
        R_sqrt = LeftMatrixSqrt(cholesky(Rₜ))
        μ₀, Σ₀_sqrt, ll₀ = rrkf_correct(m_stat, P_stat_sqrt, Hₜ, R_sqrt, yₜ, alg.nvals; compute_likelihood=compute_likelihood)
    else
        U_Pstat, s_Pstat, V_Pstat = tsvd(symmetrize_matrix(P_stat), alg.nvals)

        P_stat_sqrt = LeftMatrixSqrt(U_Pstat * Diagonal(sqrt.(s_Pstat)))
        copy(m_stat), P_stat_sqrt, 0.0
    end


    current_m, current_P_sqrt = μ₀, Σ₀_sqrt
    filter_sol = if save_all_steps
        append_step!(FilteringSolution(current_P_sqrt), previous_t, current_m, current_P_sqrt)
    else
        nothing
    end

    loglik = ll₀

    # Initialize process noise cov
    U_Q = rand_orth_mat(size(current_P_sqrt.factor)...)

    progbar = Progress(length(times) - 1; showspeed=true, enabled=show_progress)
    for (t, y) in zip(times[2:end], observations[2:end])
        current_dt = t - previous_t
        previous_t = t

        A, Q_sqrt, U_Q = discretize_lowrank(ssm.dynamics, current_dt, alg.nvals; orth_basis=U_Q)
        # Predict
        (μ⁻, Π_sqrt), (G, b, backwards_S_sqrt) = if smooth
            rrkf_predict_and_backwards_kernel(current_m, current_P_sqrt, A, Q_sqrt, alg.nvals)
        else
            rrkf_predict(current_m, current_P_sqrt, A, Q_sqrt, alg.nvals), (missing, missing, missing)
        end

        # Correct
        current_m, current_P_sqrt, ll = if !all(isnan.(y))
            Hₜ, Rₜ, yₜ = _measmod_without_nans(ssm, y)
            R_sqrt = LeftMatrixSqrt(cholesky(Rₜ))
            rrkf_correct(μ⁻, Π_sqrt, Hₜ, R_sqrt, yₜ, alg.nvals; compute_likelihood=compute_likelihood)
        else
            μ⁻, Π_sqrt, 0.0
        end

        if save_all_steps
            append_step!(filter_sol, t, current_m, current_P_sqrt, G, b, backwards_S_sqrt)
        end
        loglik += ll

        ProgressMeter.next!(progbar)
    end

    mean_loglik = loglik / length(observations)

    # Smoothing
    smoother_sol = if smooth

        rrkf_smooth(filter_sol)

    else
        nothing
    end

    estimate = Dict(
        :filter => save_all_steps ? filter_sol : (t=previous_t, μ=current_m, Σ=current_P_sqrt),
        :smoother => smoother_sol,
        :loglikelihood => mean_loglik
    )
    return estimate
end



function estimate_states(alg::EnsembleKalmanFilter, ssm, times, observations; smooth=false, compute_likelihood, save_all_steps=true, show_progress=true)

    if smooth
        error("EnKF smoothing is not implemented.")
    end

    previous_t = if length(times) == length(observations)
        times[1]
    else
        error("Lengths of times and observations don't match")
    end

    # Initialize
    # 1. compute stationary moments
    m_stat, P_stat = stationary_moments(ssm.dynamics)
    # 2. condition on first measurement
    init_ensemble, ll₀ = if !all(isnan.(observations[1]))
        Hₜ, Rₜ, yₜ = _measmod_without_nans(ssm, observations[1])
        stat_ens = rand(MvNormal(m_stat, Symmetric(P_stat)), alg.ensemble_size)
        meas_noise_dist = MvNormal(Rₜ)
        alg.correct_function(stat_ens, Hₜ, meas_noise_dist, yₜ; compute_likelihood=compute_likelihood)
    else
        stat_ens = rand(MvNormal(m_stat, Symmetric(P_stat)), alg.ensemble_size)
        stat_ens, 0.0
    end

    loglik = ll₀
    current_ensemble = init_ensemble

    filter_sol = if save_all_steps
        current_m, current_P_sqrt = ensemble_mean_sqrt_cov(current_ensemble)
        append_step!(FilteringSolution(current_P_sqrt), previous_t, current_m, current_P_sqrt)
    else
        nothing
    end


    progbar = Progress(length(times) - 1; showspeed=true, enabled=show_progress)
    for (t, y) in zip(times[2:end], observations[2:end])
        current_dt = t - previous_t
        previous_t = t

        # Discretize
        A, Q_sqrt = discretize_sqrt(ssm.dynamics, current_dt)

        # Predict
        ensemble_pred = enkf_predict(current_ensemble, A, Q_sqrt)

        # Correct
        current_ensemble, ll = if !all(isnan.(y))
            Hₜ, Rₜ, yₜ = _measmod_without_nans(ssm, y)
            obs_noise_dist = MvNormal(Rₜ)
            alg.correct_function(ensemble_pred, Hₜ, obs_noise_dist, yₜ; compute_likelihood=compute_likelihood)
        else
            ensemble_pred, 0.0
        end

        loglik += ll

        if save_all_steps
            current_m, current_P_sqrt = ensemble_mean_sqrt_cov(current_ensemble)
            append_step!(filter_sol, t, current_m, current_P_sqrt, missing, missing, missing)
        end

        ProgressMeter.next!(progbar)
    end

    mean_loglik = loglik / length(observations)

    # Smoothing (not implemented for EnKF)
    smoother_sol = nothing

    estimate = Dict(
        :filter => save_all_steps ? filter_sol : (t=previous_t, ensemble=current_ensemble),
        :smoother => smoother_sol,
        :loglikelihood => mean_loglik
    )
    return estimate
end

export estimate_states
