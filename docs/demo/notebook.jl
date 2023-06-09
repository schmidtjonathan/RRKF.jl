### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 767861bc-c886-11ed-36fb-0322b5cb2a3e
begin
	using Pkg
	Pkg.activate(".")

	using Revise
	using LinearAlgebra
	using Random
	using Distributions
	using InvertedIndices
	using StatsBase
	using KernelFunctions
	using PlutoUI
	using Plots

	using RRKF
end

# ╔═╡ 216d0060-7f69-4d21-9945-37aa5d1767d1
html"""
<style>
input[type*="range"] {
	width: 50%;
}
main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(160px, 10%);
    	padding-right: max(160px, 25%);
	}
</style>
"""

# ╔═╡ 4aeb04cd-6b51-4922-8049-610ad3fedf7d
TableOfContents()

# ╔═╡ 0a656ef8-20a5-4be6-8ba8-8fcce856db05
md"# Set up model and data"

# ╔═╡ 568bcfc2-f208-4d72-b916-55eab9c0ff94
md"## True dynamics parameters"

# ╔═╡ b59e78a8-08ff-4385-8a40-5cd181695b23
begin
	dx = 0.1                                                # Spatial grid step size
	N_x_1d = 100                                            # Number of spatial grid points
	X_grid_1d = collect(0.0:dx:(N_x_1d - 1) * dx)			# Spatial grid
	d = length(X_grid_1d)									
	dt = 0.1                                                # Temporal grid step size
	
	ν = 3/2                                                 # Smoothness of the temporal Matérn process
	ℓ_t = 2.0                                               # lengthscale of the temporal Matérn process
	ℓ_x = 0.7                                               # lengthscale of the spatial Matérn process
	σ_t = 0.4                                               # output scale of the temporal Matérn proces
	
	σ_r = 2e-2                                              # measurement noise
	D = d * round(Int, ν + 0.5)
	N_t = 500                                               # Number of temporal grid points
	simulation_grid = 0.0:dt:(N_t - 1) * dt 		        # temporal grid
end;

# ╔═╡ dc2939fa-7bee-4883-b32e-29506327da9a
begin
	K(ℓₓ) = kernelmatrix(with_lengthscale(Matern32Kernel(), ℓₓ), vec(X_grid_1d)) + 1e-7I;

	K_true = K(ℓ_x);
	true_dynamics = RRKF.build_spatiotemporal_matern_process(
	    ν,
	    ℓ_t,
	    σ_t,
	    K_true,
	)

end

# ╔═╡ 2a831aac-61e0-42ed-90c2-cb3cca9c7f64
begin
	proj(q, ν) = RRKF.projectionmatrix(d, ν, q)
end

# ╔═╡ eb889c4c-5193-40b9-b8e7-82c0e2f69fe0
md"## Draw from true dynamics model"

# ╔═╡ 23dee641-0f30-45ce-9ab3-8620ec6204ab
function simulate_linear_dynamics(
    dynamics::RRKF.LTIGaussMarkovProcess, simulation_grid
)

	μ₀, Σ₀ = RRKF.stationary_moments(dynamics)
    x = rand(MvNormal(μ₀, Symmetric(Σ₀)))

	trajectory = Vector{Float64}[x]

	prev_t = simulation_grid[1]
    for t in simulation_grid[2:end]
		dt = t - prev_t
		A, Q = RRKF.discretize(dynamics, dt)
		push!(trajectory, rand(MvNormal(A * trajectory[end], Symmetric(Q))))
		prev_t = t
    end
    return vecvec2mat(trajectory)
end


# ╔═╡ c791dde6-dd43-41f8-960e-f44ce2fd5c04
Random.seed!(1234)

# ╔═╡ d88f5d96-c749-4652-9eb2-ab2326127fcb
prior_draw = simulate_linear_dynamics(
    true_dynamics, simulation_grid
);

# ╔═╡ f7503c2b-7520-470c-978f-872321109a89
function imshow(A; kwargs...)
	heatmap(A'; yflip=true, kwargs...)
end

# ╔═╡ e3e852ca-3f0a-4f12-98de-bef5bf379060
materndraw = prior_draw * proj(0, ν)';

# ╔═╡ fcf8c433-896d-4bdd-8558-eff4d727c347
begin
	ground_truth =  materndraw
end;

# ╔═╡ e5aae12c-b10b-4596-8cf3-d9800f4b3bfc
begin
	smplx = rand(1:size(ground_truth, 2))
	gt_hm = Plots.heatmap(simulation_grid, X_grid_1d, ground_truth', xlabel="t", ylabel="x", yflip=true, title="ground truth")
	hline!([X_grid_1d[smplx]], color=1, label="", lw=2)
	gt_l = Plots.plot(ground_truth[:, smplx], label="x = $(X_grid_1d[smplx])")
	Plots.plot(gt_hm, gt_l, layout=(2, 1))
end

# ╔═╡ 6a055d94-1a29-4958-987d-67310b07a656
md"## Generate noisy data from draw"

# ╔═╡ 3bcb05aa-2fc5-476a-9e99-40feebfd791d
size(ground_truth)

# ╔═╡ 2b01d210-c57b-4f2b-9b4f-4e6d0ff379e9
num_measurements = 40

# ╔═╡ 9ad18d64-9dac-4f20-b102-76ae84192dae
dim_measurements = 25

# ╔═╡ 807859c1-937f-41f2-a7b9-0ff6357a2b97
measurement_times_idcs = sort(sample(axes(ground_truth, 1)[2:end], num_measurements, replace=false))

# ╔═╡ 6feefb4d-3426-49bb-9e42-fe45668ab949
measurement_times = simulation_grid[measurement_times_idcs]

# ╔═╡ f6212cc3-677d-486a-b82b-5f608c9d2a21
measurement_locations_idcs = [sample(axes(ground_truth, 2), dim_measurements, replace=false) for i in measurement_times_idcs]

# ╔═╡ 08bdf95a-f249-46ab-bec7-3e8b80f324f7
measurement_locations = [X_grid_1d[ix] for ix in measurement_locations_idcs]

# ╔═╡ 99e565fb-dc45-4927-996b-d524ab58660b
begin
	data_mat = copy(ground_truth)
	data_mat[Not(measurement_times_idcs), :] .= NaN64
	for (i,tp) in enumerate(measurement_times_idcs)
		data_mat[tp, Not(measurement_locations_idcs[i])] .= NaN64
	end
	noisy_data_mat = data_mat .+ rand(MvNormal(zeros(N_x_1d), σ_r^2 * I(N_x_1d)), N_t)'
end;

# ╔═╡ e97a2983-182d-4791-8e92-e91a60dd5e55
data_mat[.~isnan.(data_mat)]

# ╔═╡ 3e2f59cb-a1b5-4559-ae41-2694ef7c38e9
noisy_data_mat[.~isnan.(noisy_data_mat)]

# ╔═╡ 7384a058-3224-4444-a95a-a26b82318c50
@show size(data_mat) size(ground_truth)

# ╔═╡ 578182a1-46cd-4bde-b0da-9588d4f152c6


# ╔═╡ 207fe8e8-a04f-47c9-9df8-1ec08f8c887f
begin
	noisy_data_hm = Plots.heatmap(simulation_grid, X_grid_1d, noisy_data_mat', xlabel="t", ylabel="x", yflip=true, title="noisy data")
end

# ╔═╡ de3e007a-773e-473b-8edb-3b287d0fddd1
noisy_data = [noisy_data_mat[t, :] for t in axes(noisy_data_mat, 1)];

# ╔═╡ 9148a285-ed28-4b31-8e5f-795506de746e
@assert length([d for d in noisy_data if !all(isnan.(d))]) == num_measurements

# ╔═╡ e44460ad-0e90-4e63-b5ac-1f04cab7ab24
# Magic code that is needed for plotting. Can be ignored
begin
	dat_idcs_for_x = [findfirst(yind .== smplx) for yind in measurement_locations_idcs]
	times_at_which_x_is_measured = findall(!isnothing, dat_idcs_for_x)
	data_at_those_times_at_x = [noisy_data[t][s] for (t, s) in zip(times_at_which_x_is_measured, dat_idcs_for_x[times_at_which_x_is_measured])]
	time_points_at_those_times = measurement_times[times_at_which_x_is_measured]
end;

# ╔═╡ 8a9f8bd8-0997-4228-b16d-da9c2fc5e7bc
md"# Set up prior model"

# ╔═╡ c3e94ae2-1605-42d3-be58-ae5d72ea20a7
md"**Here we introduce an artificial model mismatch by approximating the draw from a Matern 3/2 using a Matern 1/2 prior**"

# ╔═╡ 673da733-6183-4dda-a8f2-04c49b6d76d8
model_ν = 1/2

# ╔═╡ 56e9798c-ffa5-41c8-ae4c-ee6a050c72e0
model_D = d * round(Int, model_ν + 0.5)

# ╔═╡ 8f72645b-62a2-45b6-95e1-0fd0a6799e81
sqrt_diffusion = RRKF.LeftMatrixSqrt(cholesky(K_true))

# ╔═╡ 1be2e11a-7cb0-4f53-b3b9-1626dd473a11
prior_dynamics = RRKF.build_spatiotemporal_matern_process(
	model_ν, ℓ_t, σ_t, sqrt_diffusion
)

# ╔═╡ 2bb2a833-f721-4d16-90bc-8b87e5b416c5
ssm = RRKF.StateSpaceModel(
	prior_dynamics,
	proj(0, model_ν),
	σ_r^2 * Matrix{Float64}(I, d, d)
)

# ╔═╡ d99d977d-46d9-4584-8bfa-bc8d6cc57760
sqrtkf_run = RRKF.estimate_states(
	RRKF.SqrtKalmanFilter(),
	ssm,
	simulation_grid,
	noisy_data;
	smooth=true,
	save_all_steps = true,
	show_progress=false,
	compute_likelihood=true,
)

# ╔═╡ bc8b8a97-8314-4031-a2cd-35a52be2c9b4
sqrtkf_filter_sol = sqrtkf_run[:filter]

# ╔═╡ 464489e4-98dd-46ec-ab26-789c584e9947
sqrtkf_smoother_sol = sqrtkf_run[:smoother]

# ╔═╡ 6ae408e7-e501-44c6-9d0e-d8c2def2c774
sqrtkf_filter_estimate = RRKF.means(sqrtkf_filter_sol) * proj(0, model_ν)';

# ╔═╡ 635760d2-b1e2-4f82-be0d-2c20991fb8bc
sqrtkf_filter_estimate_std = RRKF.stds(sqrtkf_filter_sol) * proj(0, model_ν)';

# ╔═╡ 45bdabae-b402-4fe9-8e09-b656907ab333
md"#### Spectrum of the KF covariance estimate"

# ╔═╡ ad0140c0-65c7-4643-a922-dbe7e84891dc
begin
	spectrum_of_problem = eigvals(Matrix(sqrtkf_filter_sol.Σ[end]), sortby=l->-l)
	normalized_cumulative_spectrum_of_problem = cumsum(spectrum_of_problem)/sum(spectrum_of_problem)
	Plots.scatter(normalized_cumulative_spectrum_of_problem, ylim=(-0.05, 1.05), markersize=2, label="cumulative normalized eigenvalues")
end

# ╔═╡ 4b6db141-6269-4563-bf4e-9409891b3c83
default_r = findfirst(normalized_cumulative_spectrum_of_problem .>= 0.95)

# ╔═╡ 4da62ed8-bcca-4c8b-b4ee-fa292249bb39
md"# Have a look at the solutions"

# ╔═╡ 55bb18e6-7d2b-4e82-9af7-41c886a1cff2
md"## Filtering"

# ╔═╡ f2df01eb-68da-49ff-8969-31898fe7c836
md"""
#### Play around with `r` and the number of DLR integrator steps

For the plot below ⤵
"""

# ╔═╡ c39dec6d-7137-414f-bd3b-32f769cd8530
begin
	slider_r = @bind r PlutoUI.Slider(1:model_D, default=default_r, show_value=true)

	md"""
	Choose r: $(slider_r)
	"""
end

# ╔═╡ 13525fc8-63c8-4a84-a1ec-46e607ca32ef
rrkf_run = RRKF.estimate_states(
	RRKF.RankReducedKalmanFilter(r, 1),
	ssm,
	simulation_grid,
	noisy_data;
	smooth=true,
	save_all_steps = true,
	show_progress=false,
	compute_likelihood=true,
)

# ╔═╡ 2e76f207-d43d-451a-8b95-81912ccbc771
rrkf_filter_sol = rrkf_run[:filter]

# ╔═╡ 5e8e2ac5-f389-4571-bb5c-41422af4bde6
rrkf_filter_estimate = RRKF.means(rrkf_filter_sol) * proj(0, model_ν)';

# ╔═╡ 70b7fdfd-bc0c-40e6-96bf-0b25fd9f553a
rrkf_filter_estimate_std = RRKF.stds(rrkf_filter_sol) * proj(0, model_ν)';

# ╔═╡ 45861c56-e5ae-4c4a-bdc8-fbb9d784c339
rrkf_smoother_sol = rrkf_run[:smoother]

# ╔═╡ 820d936b-ed20-47a1-a871-153456803a46
enkf_run = RRKF.estimate_states(
	RRKF.EnsembleKalmanFilter(r, RRKF.enkf_correct),
	ssm,
	simulation_grid,
	noisy_data;
	smooth=false,
	save_all_steps = true,
	show_progress=false,
	compute_likelihood=false,
)

# ╔═╡ 16f857d9-10c6-4837-9e40-7d20eb2ae57e
enkf_filter_sol = enkf_run[:filter]

# ╔═╡ dd1118ec-8635-4931-884b-8983dd1925aa
enkf_filter_estimate = RRKF.means(enkf_filter_sol) * proj(0, model_ν)';

# ╔═╡ 9d5b2ef4-71ca-4df6-8f16-5304383eb71e
enkf_filter_estimate_std = RRKF.stds(enkf_filter_sol) * proj(0, model_ν)';

# ╔═╡ 2cd19c5e-ca24-4300-aea0-4185112f332e
md"## Smoothing"

# ╔═╡ f1f87aff-a1bd-40f8-b842-08d8ee4c006f
sqrtkf_smoother_estimate = RRKF.means(sqrtkf_smoother_sol) * proj(0, model_ν)';

# ╔═╡ 44df41aa-7442-4c3d-b808-0a429e9c65cc
sqrtkf_smoother_estimate_std = RRKF.stds(sqrtkf_smoother_sol) * proj(0, model_ν)';

# ╔═╡ 6d954132-c2f9-4071-856f-d12b08cb61ce
rrkf_smoother_estimate = RRKF.means(rrkf_smoother_sol) * proj(0, model_ν)';

# ╔═╡ 47a7a38f-f8f7-454e-a624-60d056b034b6
rrkf_smoother_estimate_std = RRKF.stds(rrkf_smoother_sol) * proj(0, model_ν)';

# ╔═╡ 597592bf-1099-451d-8928-a787af4ddc14
rrkf_smoother_sol[1]

# ╔═╡ 39d65305-48fa-488c-9459-e22b8deac668
md"# Plotting code"

# ╔═╡ 2c3b8f7e-e868-497a-8105-a5a0bc8143f1
begin
	sqrtkf_hm = Plots.heatmap(simulation_grid, X_grid_1d, sqrtkf_filter_estimate', xlabel="t", ylabel="x", yflip=true, title="SqrtKF estimate")
	# scatter!(measurement_times, vecvec2mat(measurement_locations), label="", markersize=1, color=:black, alpha=0.8)
	sqrtkf_hm_std = Plots.heatmap(simulation_grid, X_grid_1d, sqrtkf_filter_estimate_std', xlabel="t", ylabel="x", yflip=true, title="SqrtKF std", c=:haline)
	# scatter!(measurement_times, vecvec2mat(measurement_locations), label="", markersize=1, color=:black, alpha=0.8)
	sqrtkf_gt_vs_filt = Plots.plot(sqrtkf_hm, sqrtkf_hm_std, gt_hm, layout=(1, 3))
	sqrtkf_l = Plots.plot(simulation_grid, ground_truth[:, smplx], label="x = $(round(X_grid_1d[smplx]; digits=2))", xlabel="t", ylabel="x_$smplx", ylim=(-1, 1))
	Plots.plot!(simulation_grid, sqrtkf_filter_estimate[:, smplx], ribbon=2*sqrtkf_filter_estimate_std[:, smplx], label="filter")
	Plots.scatter!(time_points_at_those_times, data_at_those_times_at_x, color=:black, markersize=2, label="y")
	sqrtkf_plot = Plots.plot(sqrtkf_gt_vs_filt, sqrtkf_l, layout=(2, 1), size=(1000, 400))
end;

# ╔═╡ 5dda57e9-1e51-4d57-b464-4acff23fe4c3
Plots.plot(sqrtkf_plot)

# ╔═╡ 2d0012be-e8ee-452c-8ce8-6758a5e55a8d
begin
	rrkf_hm = Plots.heatmap(simulation_grid, X_grid_1d, rrkf_filter_estimate', xlabel="t", ylabel="x", yflip=true, title="RRKF($r) estimate")
	# scatter!(measurement_times, vecvec2mat(measurement_locations), label="", markersize=1, color=:black, alpha=0.8)
	rrkf_hm_std = Plots.heatmap(simulation_grid, X_grid_1d, rrkf_filter_estimate_std', xlabel="t", ylabel="x", yflip=true, title="RRKF($r) std", c=:haline)
	# scatter!(measurement_times, vecvec2mat(measurement_locations), label="", markersize=1, color=:black, alpha=0.8)
	rrkf_gt_vs_filt = Plots.plot(rrkf_hm, rrkf_hm_std, gt_hm, layout=(1, 3))
	rrkf_l = Plots.plot(simulation_grid, ground_truth[:, smplx], label="x = $(round(X_grid_1d[smplx]; digits=2))", xlabel="t", ylabel="x_$smplx", ylim=(-1, 1))
	Plots.plot!(simulation_grid, rrkf_filter_estimate[:, smplx], ribbon=2*rrkf_filter_estimate_std[:, smplx], label="filter")
	Plots.scatter!(time_points_at_those_times, data_at_those_times_at_x, color=:black, markersize=2, label="y")
	rrkf_plot = Plots.plot(rrkf_gt_vs_filt, rrkf_l, layout=(2, 1), size=(1000, 400))
end;

# ╔═╡ 2ed2114e-2a76-42cd-b8bd-39066bf5c033
Plots.plot(rrkf_plot)

# ╔═╡ acf86e77-32db-4298-ade2-a0561f980b66
begin
	enkf_hm = Plots.heatmap(simulation_grid, X_grid_1d, enkf_filter_estimate', xlabel="t", ylabel="x", yflip=true, title="EnKF($r) estimate")
	# scatter!(measurement_times, vecvec2mat(measurement_locations), label="", markersize=1, color=:black, alpha=0.8)
	enkf_hm_std = Plots.heatmap(simulation_grid, X_grid_1d, enkf_filter_estimate_std', xlabel="t", ylabel="x", yflip=true, title="EnKF($r) std", c=:haline)
	# scatter!(measurement_times, vecvec2mat(measurement_locations), label="", markersize=1, color=:black, alpha=0.8)
	enkf_gt_vs_filt = Plots.plot(enkf_hm, enkf_hm_std, gt_hm, layout=(1, 3))
	enkf_l = Plots.plot(simulation_grid, ground_truth[:, smplx], label="x = $(round(X_grid_1d[smplx]; digits=2))", xlabel="t", ylabel="x_$smplx", ylim=(-1, 1))
	Plots.plot!(simulation_grid, enkf_filter_estimate[:, smplx], ribbon=2*enkf_filter_estimate_std[:, smplx], label="filter")
	Plots.scatter!(time_points_at_those_times, data_at_those_times_at_x, color=:black, markersize=2, label="y")
	enkf_plot = Plots.plot(enkf_gt_vs_filt, enkf_l, layout=(2, 1), size=(1000, 400))
end;

# ╔═╡ bbbfd395-aa0d-49db-ba90-24bcfb473980
Plots.plot(enkf_plot)

# ╔═╡ e578a5f0-0b59-42b3-8ece-8d7f5704b21e
begin
	sqrtkf_smoother_hm = Plots.heatmap(simulation_grid, X_grid_1d, sqrtkf_smoother_estimate', xlabel="t", ylabel="x", yflip=true, title="SqrtKF smoother estimate")
	# scatter!(measurement_times, vecvec2mat(measurement_locations), label="", markersize=1, color=:black, alpha=0.8)
	sqrtkf_smoother_hm_std = Plots.heatmap(simulation_grid, X_grid_1d, sqrtkf_smoother_estimate_std', xlabel="t", ylabel="x", yflip=true, title="SqrtKF smoother std", c=:haline)
	# scatter!(measurement_times, vecvec2mat(measurement_locations), label="", markersize=1, color=:black, alpha=0.8)
	sqrtkf_smoother_gt_vs_filt = Plots.plot(sqrtkf_smoother_hm, sqrtkf_smoother_hm_std, gt_hm, layout=(1, 3))
	sqrtkf_smoother_l = Plots.plot(simulation_grid, ground_truth[:, smplx], label="x = $(round(X_grid_1d[smplx]; digits=2))", xlabel="t", ylabel="x_$smplx", ylim=(-1, 1))
	Plots.plot!(simulation_grid, sqrtkf_smoother_estimate[:, smplx], ribbon=2*sqrtkf_smoother_estimate_std[:, smplx], label="smoother")
	Plots.scatter!(time_points_at_those_times, data_at_those_times_at_x, color=:black, markersize=2, label="y")
	sqrtkf_smoother_plot = Plots.plot(sqrtkf_smoother_gt_vs_filt, sqrtkf_smoother_l, layout=(2, 1), size=(1000, 400))
end;

# ╔═╡ d4f16214-9181-4d81-8637-5d67abf42277
Plots.plot(sqrtkf_smoother_plot)

# ╔═╡ 1940ade5-7a8d-4527-af16-25feb6cc3368
begin
	rrkf_smoother_hm = Plots.heatmap(simulation_grid, X_grid_1d, rrkf_smoother_estimate', xlabel="t", ylabel="x", yflip=true, title="rrkf smoother estimate")
	# scatter!(measurement_times, vecvec2mat(measurement_locations), label="", markersize=1, color=:black, alpha=0.8)
	rrkf_smoother_hm_std = Plots.heatmap(simulation_grid, X_grid_1d, rrkf_smoother_estimate_std', xlabel="t", ylabel="x", yflip=true, title="rrkf smoother std", c=:haline)
	# scatter!(measurement_times, vecvec2mat(measurement_locations), label="", markersize=1, color=:black, alpha=0.8)
	rrkf_smoother_gt_vs_filt = Plots.plot(rrkf_smoother_hm, rrkf_smoother_hm_std, gt_hm, layout=(1, 3))
	rrkf_smoother_l = Plots.plot(simulation_grid, ground_truth[:, smplx], label="x = $(round(X_grid_1d[smplx]; digits=2))", xlabel="t", ylabel="x_$smplx", )
	Plots.plot!(simulation_grid, rrkf_smoother_estimate[:, smplx], ribbon=2*rrkf_smoother_estimate_std[:, smplx], label="smoother", ylim=(-1, 1))
	Plots.scatter!(time_points_at_those_times, data_at_those_times_at_x, color=:black, markersize=2, label="y")
	rrkf_smoother_plot = Plots.plot(rrkf_smoother_gt_vs_filt, rrkf_smoother_l, layout=(2, 1), size=(1000, 400))
end;

# ╔═╡ cd485c67-b6f9-4f1e-8f81-3fc06a001e7a
Plots.plot(rrkf_smoother_plot)

# ╔═╡ Cell order:
# ╟─216d0060-7f69-4d21-9945-37aa5d1767d1
# ╠═767861bc-c886-11ed-36fb-0322b5cb2a3e
# ╟─4aeb04cd-6b51-4922-8049-610ad3fedf7d
# ╟─0a656ef8-20a5-4be6-8ba8-8fcce856db05
# ╟─568bcfc2-f208-4d72-b916-55eab9c0ff94
# ╠═b59e78a8-08ff-4385-8a40-5cd181695b23
# ╠═dc2939fa-7bee-4883-b32e-29506327da9a
# ╠═2a831aac-61e0-42ed-90c2-cb3cca9c7f64
# ╟─eb889c4c-5193-40b9-b8e7-82c0e2f69fe0
# ╠═23dee641-0f30-45ce-9ab3-8620ec6204ab
# ╠═c791dde6-dd43-41f8-960e-f44ce2fd5c04
# ╠═d88f5d96-c749-4652-9eb2-ab2326127fcb
# ╠═f7503c2b-7520-470c-978f-872321109a89
# ╠═e3e852ca-3f0a-4f12-98de-bef5bf379060
# ╠═fcf8c433-896d-4bdd-8558-eff4d727c347
# ╠═e5aae12c-b10b-4596-8cf3-d9800f4b3bfc
# ╟─6a055d94-1a29-4958-987d-67310b07a656
# ╠═3bcb05aa-2fc5-476a-9e99-40feebfd791d
# ╠═2b01d210-c57b-4f2b-9b4f-4e6d0ff379e9
# ╠═9ad18d64-9dac-4f20-b102-76ae84192dae
# ╠═807859c1-937f-41f2-a7b9-0ff6357a2b97
# ╠═6feefb4d-3426-49bb-9e42-fe45668ab949
# ╠═f6212cc3-677d-486a-b82b-5f608c9d2a21
# ╠═08bdf95a-f249-46ab-bec7-3e8b80f324f7
# ╠═99e565fb-dc45-4927-996b-d524ab58660b
# ╠═e97a2983-182d-4791-8e92-e91a60dd5e55
# ╠═3e2f59cb-a1b5-4559-ae41-2694ef7c38e9
# ╠═7384a058-3224-4444-a95a-a26b82318c50
# ╠═578182a1-46cd-4bde-b0da-9588d4f152c6
# ╠═207fe8e8-a04f-47c9-9df8-1ec08f8c887f
# ╠═de3e007a-773e-473b-8edb-3b287d0fddd1
# ╠═9148a285-ed28-4b31-8e5f-795506de746e
# ╟─e44460ad-0e90-4e63-b5ac-1f04cab7ab24
# ╟─8a9f8bd8-0997-4228-b16d-da9c2fc5e7bc
# ╟─c3e94ae2-1605-42d3-be58-ae5d72ea20a7
# ╠═673da733-6183-4dda-a8f2-04c49b6d76d8
# ╠═56e9798c-ffa5-41c8-ae4c-ee6a050c72e0
# ╠═8f72645b-62a2-45b6-95e1-0fd0a6799e81
# ╠═1be2e11a-7cb0-4f53-b3b9-1626dd473a11
# ╠═2bb2a833-f721-4d16-90bc-8b87e5b416c5
# ╠═d99d977d-46d9-4584-8bfa-bc8d6cc57760
# ╠═bc8b8a97-8314-4031-a2cd-35a52be2c9b4
# ╠═464489e4-98dd-46ec-ab26-789c584e9947
# ╠═6ae408e7-e501-44c6-9d0e-d8c2def2c774
# ╠═635760d2-b1e2-4f82-be0d-2c20991fb8bc
# ╟─45bdabae-b402-4fe9-8e09-b656907ab333
# ╠═ad0140c0-65c7-4643-a922-dbe7e84891dc
# ╠═4b6db141-6269-4563-bf4e-9409891b3c83
# ╠═13525fc8-63c8-4a84-a1ec-46e607ca32ef
# ╠═2e76f207-d43d-451a-8b95-81912ccbc771
# ╠═45861c56-e5ae-4c4a-bdc8-fbb9d784c339
# ╠═5e8e2ac5-f389-4571-bb5c-41422af4bde6
# ╠═70b7fdfd-bc0c-40e6-96bf-0b25fd9f553a
# ╠═820d936b-ed20-47a1-a871-153456803a46
# ╠═16f857d9-10c6-4837-9e40-7d20eb2ae57e
# ╠═dd1118ec-8635-4931-884b-8983dd1925aa
# ╠═9d5b2ef4-71ca-4df6-8f16-5304383eb71e
# ╟─4da62ed8-bcca-4c8b-b4ee-fa292249bb39
# ╟─55bb18e6-7d2b-4e82-9af7-41c886a1cff2
# ╟─5dda57e9-1e51-4d57-b464-4acff23fe4c3
# ╟─f2df01eb-68da-49ff-8969-31898fe7c836
# ╟─c39dec6d-7137-414f-bd3b-32f769cd8530
# ╟─2ed2114e-2a76-42cd-b8bd-39066bf5c033
# ╟─bbbfd395-aa0d-49db-ba90-24bcfb473980
# ╟─2cd19c5e-ca24-4300-aea0-4185112f332e
# ╠═f1f87aff-a1bd-40f8-b842-08d8ee4c006f
# ╠═44df41aa-7442-4c3d-b808-0a429e9c65cc
# ╠═6d954132-c2f9-4071-856f-d12b08cb61ce
# ╠═47a7a38f-f8f7-454e-a624-60d056b034b6
# ╠═597592bf-1099-451d-8928-a787af4ddc14
# ╠═d4f16214-9181-4d81-8637-5d67abf42277
# ╠═cd485c67-b6f9-4f1e-8f81-3fc06a001e7a
# ╟─39d65305-48fa-488c-9459-e22b8deac668
# ╠═2c3b8f7e-e868-497a-8105-a5a0bc8143f1
# ╠═2d0012be-e8ee-452c-8ce8-6758a5e55a8d
# ╠═acf86e77-32db-4298-ade2-a0561f980b66
# ╠═e578a5f0-0b59-42b3-8ece-8d7f5704b21e
# ╠═1940ade5-7a8d-4527-af16-25feb6cc3368
