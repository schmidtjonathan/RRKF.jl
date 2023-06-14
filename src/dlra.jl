# DLR FOR SYMMETRIC LYAPUNOV EQUATIONS Ṗ = FP + PF' + C


function K_step_lyapunov(K, drift_matrix, Vₙ, diffusion_matrix, t)
    return drift_matrix * K + K * (Vₙ' * drift_matrix' * Vₙ) + diffusion_matrix * Vₙ
end

function S_step_lyapunov(S, drift_matrix, Û, diffusion_matrix, t)
    S_Ut_Ft_U = S * Û' * drift_matrix' * Û
    return S_Ut_Ft_U' + S_Ut_Ft_U + Û' * diffusion_matrix * Û
end


function closed_form_K_step_lyapunov(drift_matrix, V, diffusion_matrix, Kₙ, h)
    d_select = size(drift_matrix, 1)
    Mexp = exp(h * [drift_matrix (diffusion_matrix*V); zero(V') (-V'*drift_matrix'*V)])
    Mexp_11 = Mexp[1:d_select, 1:d_select]
    Mexp_12 = Mexp[1:d_select, d_select+1:end]
    Mexp_22 = Mexp[d_select+1:end, d_select+1:end]
    Knext = (Mexp_11 * Kₙ + Mexp_12) / Mexp_22
    return Knext
end


function closed_form_S_step_lyapunov(drift_matrix, Û, diffusion_matrix, Ŝₙ, h)
    hAₛ = h * (Û' * (drift_matrix * Û))
    hPₛ = h * (Û' * (diffusion_matrix * Û))

    d_select = size(hAₛ, 1)
    Mexp = exp([hAₛ hPₛ; zero(hAₛ) -hAₛ'])
    Mexp_11 = Mexp[1:d_select, 1:d_select]
    Mexp_12 = Mexp[1:d_select, d_select+1:end]
    Snext = (Mexp_11 * Ŝₙ + Mexp_12) * Mexp_11'
    return Snext
end


function solve_lyapunov_dlr(
    proc::LTIGaussMarkovProcess,
    Y::LeftMatrixSqrt,
    r₀,
    t_span,
    num_steps,
)
    QR_L = qr(Y.factor)
    Q_L = Matrix(QR_L.Q)
    R_L = Matrix(QR_L.R)
    Y_tpl = (Q_L, symmetrize_matrix(R_L * R_L'))
    solve_lyapunov_dlr(proc, Y_tpl, r₀, t_span, num_steps)
end


function solve_lyapunov_dlr(proc::LTIGaussMarkovProcess, Y::Tuple, r₀, dt, num_steps)
    splitting_integrator_steps, h = if num_steps == 1
        [dt], dt
    else
        _rg = LinRange(0.0, dt, num_steps + 1)
        _st = _rg[2] - _rg[1]
        _rg[2:end], _st
    end

    @assert size(Y[1], 2) == size(Y[2], 1) == size(Y[2], 2) == r₀
    small_matexp = exp(proc.drift_matrix_1d * h)
    eAh = kronecker(I(wiener_process_dimension(proc)), small_matexp)

    phi_factor = kronecker(
        I(wiener_process_dimension(proc)),
        (small_matexp - I) / (proc.drift_matrix_1d * h),
    )
    current_Y = Y

    @assert length(splitting_integrator_steps) == num_steps
    for tn in splitting_integrator_steps
        Uₙ, Sₙ = current_Y
        Vₙ = Uₙ

        # K STEP
        Kₙ = Uₙ * Sₙ

        K_next = if iszero(Sₙ)
            h * phi_factor * (proc.diffusion_matrix * Vₙ)
        else
            eAh * Kₙ +
            h *
            phi_factor *
            (Kₙ * (Vₙ' * (proc.drift_matrix' * Vₙ)) + proc.diffusion_matrix * Vₙ)
        end

        qr_K_next = qr!(K_next)
        Û = Matrix(qr_K_next.Q)
        M̂ = Û' * Uₙ

        N̂ = M̂

        Ŝₙ = M̂ * Sₙ * N̂'
        S_next = closed_form_S_step_lyapunov(
            proc.drift_matrix,
            Û,
            proc.diffusion_matrix,
            Ŝₙ,
            h,
        )

        current_Y = (Û, symmetrize_matrix(S_next))
    end
    chol_S = cholesky(Symmetric(current_Y[2], :L)).L
    return current_Y[1], chol_S
end
