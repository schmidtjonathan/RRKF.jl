struct LTIGaussMarkovProcess{amT,bmT,iamT,ibmT,wdT,icmT}
    drift_matrix_1d::amT
    dispersion_matrix_1d::bmT
    drift_matrix::iamT
    dispersion_matrix::ibmT
    wiener_diffusion::wdT
    diffusion_matrix::icmT
end


function build_spatiotemporal_matern_process(ν, ℓₜ, σₜ, spatial_kernelmatrix)
    drift_matrix_1d, dispersion_matrix_1d = matern_1d_LTISDE(ν, ℓₜ, σₜ)
    wd = size(spatial_kernelmatrix, 1)
    drift_matrix = kronecker(I(wd), drift_matrix_1d)
    dispersion_matrix = kronecker(I(wd), dispersion_matrix_1d)
    diffusion_matrix = dispersion_matrix * Matrix(spatial_kernelmatrix) * dispersion_matrix'
    return LTIGaussMarkovProcess(
        drift_matrix_1d,
        dispersion_matrix_1d,
        drift_matrix,
        dispersion_matrix,
        spatial_kernelmatrix,
        diffusion_matrix,
    )
end


function wiener_process_dimension(proc::LTIGaussMarkovProcess)
    return size(proc.wiener_diffusion, 1)
end

function num_derivatives(proc::LTIGaussMarkovProcess)
    return size(proc.drift_matrix_1d, 1) - 1
end

function state_dimension(proc::LTIGaussMarkovProcess)
    return wiener_process_dimension(proc) * (num_derivatives(proc) + 1)
end


function matern_1d_LTISDE(smoothness, lengthscale, output_scale)
    # Särkka, S: Applied SDEs. Section 12.3
    ν = smoothness
    @assert ν % 1.0 == 0.5
    ℓ = lengthscale
    D = round(Int, ν + 0.5)
    λ = sqrt(2ν) / ℓ

    drift = diagm(1 => ones(D - 1))
    for i = 0:1:D-1
        aᵢ = binomial(D, i)
        drift[end, i+1] = -aᵢ * λ^(D - i)
    end

    dispersion = zeros(D)
    diffusion_coeff =
        output_scale^2 * ((factorial(D - 1)^2) / (factorial(2 * D - 2))) * (2λ)^(2 * D - 1)
    dispersion[end] = diffusion_coeff

    return drift, dispersion
end


function stationary_moments(proc::LTIGaussMarkovProcess)
    Σ_stationary_1d =
        lyapc(proc.drift_matrix_1d, proc.dispersion_matrix_1d * proc.dispersion_matrix_1d')
    # https://aaltodoc.aalto.fi/bitstream/handle/123456789/19842/isbn9789526067117.pdf?sequence=1&isAllowed=y
    #                   /--------------------------\ <- around Eq. (4.17) in Arno Solin's thesis --^
    full_diffusion_matrix = Symmetric(symmetrize_matrix(Matrix(proc.wiener_diffusion)))
    Σ_stationary = kronecker(full_diffusion_matrix, Σ_stationary_1d)

    μ_stationary = zeros(size(Σ_stationary, 1))
    return μ_stationary, Σ_stationary
end


function discretize(proc::LTIGaussMarkovProcess, dt)
    d = size(proc.drift_matrix_1d, 1)
    M = [
        proc.drift_matrix_1d proc.dispersion_matrix_1d*proc.dispersion_matrix_1d'
        zero(proc.drift_matrix_1d) -proc.drift_matrix_1d'
    ]
    Mexp = exp(dt * M)
    A_breve = Mexp[1:d, 1:d]
    Q_breve = Mexp[1:d, d+1:end] * A_breve'
    Q_breve = symmetrize_matrix(Q_breve)

    A = kronecker(I(wiener_process_dimension(proc)), A_breve)
    full_diffusion_matrix = Symmetric(symmetrize_matrix(Matrix(proc.wiener_diffusion)))
    Q = kronecker(full_diffusion_matrix, Q_breve)

    return A, Q
end


function discretize_sqrt(proc::LTIGaussMarkovProcess, dt)
    diff_sqrt = if proc.wiener_diffusion isa LeftMatrixSqrt
        proc.wiener_diffusion.factor
    else
        @warn "Called discretize_sqrt without square-root factor of diffusion provided. Falling back to computing Cholesky factor in each call, which is very inefficient."
        cholesky(proc.wiener_diffusion)
    end

    d = size(proc.drift_matrix_1d, 1)
    M = [
        proc.drift_matrix_1d proc.dispersion_matrix_1d*proc.dispersion_matrix_1d'
        zero(proc.drift_matrix_1d) -proc.drift_matrix_1d'
    ]
    Mexp = exp(dt * M)
    A_breve = Mexp[1:d, 1:d]
    Q_breve = Mexp[1:d, d+1:end] * A_breve'
    Q_breve = symmetrize_matrix(Q_breve)
    Q_breve_sqrt = cholesky(Symmetric(Q_breve, :L)).L

    A = kronecker(I(wiener_process_dimension(proc)), A_breve)
    Q_sqrt = LeftMatrixSqrt(kron(diff_sqrt, Q_breve_sqrt))

    return A, Q_sqrt
end


function discretize_lowrank(
    proc::LTIGaussMarkovProcess,
    dt,
    r;
    orth_basis = nothing,
    num_dlra_steps = 1,
)

    U0 = if isnothing(orth_basis)
        rand_orth_mat(state_dimension(proc), r)
    else
        orth_basis
    end
    Q0 = (U0, Diagonal(zeros(size(U0, 2))))
    U_Q, D_Q = solve_lyapunov_dlr(proc, Q0, r, dt, num_dlra_steps)
    Q_sqrt = LeftMatrixSqrt(U_Q * D_Q)

    A_breve = exp(dt * proc.drift_matrix_1d)
    A = kronecker(I(wiener_process_dimension(proc)), A_breve)

    return A, Q_sqrt, U_Q

end


function projectionmatrix(proc::LTIGaussMarkovProcess, derivative)
    dimension = wiener_process_dimension(proc)
    D = num_derivatives(proc) + 1
    return kronecker(I(dimension), [i == (derivative + 1) ? 1.0 : 0.0 for i = 1:D]')
end


function projectionmatrix(dimension::Int64, smoothness::Float64, derivative::Int64)
    @assert smoothness % 1.0 == 0.5
    D = round(Int, smoothness + 0.5)
    return kronecker(I(dimension), [i == (derivative + 1) ? 1.0 : 0.0 for i = 1:D]')
end
