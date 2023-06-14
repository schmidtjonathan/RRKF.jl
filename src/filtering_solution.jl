abstract type AbstractKalmanSolution{CT} end

struct FilteringSolution{CT} <: AbstractKalmanSolution{CT}
    t::Vector{Float64}
    μ::Vector{Vector{Float64}}
    Σ::Vector{CT}

    backwards_transitions::Vector{Tuple}

    FilteringSolution{CT}() where {CT} =
        new{CT}(Float64[], Vector{Float64}[], CT[], Tuple[])
    FilteringSolution(cov::CT) where {CT} =
        new{typeof(cov)}(Float64[], Vector{Float64}[], typeof(cov)[], Tuple[])
end

struct SmoothingSolution{CT} <: AbstractKalmanSolution{CT}
    t::Vector{Float64}
    μ::Vector{Vector{Float64}}
    Σ::Vector{CT}

    SmoothingSolution{CT}() where {CT} = new{CT}(Float64[], Vector{Float64}[], CT[])
end


function SmoothingSolution(filter_sol::FilteringSolution)
    return smooth(filter_sol)
end


function append_step!(sol::SmoothingSolution, t, μ, Σ)
    Base.pushfirst!(sol.t, t)
    Base.pushfirst!(sol.μ, copy(μ))
    Base.pushfirst!(sol.Σ, copy(Σ))

    return sol
end


function append_step!(
    sol::FilteringSolution,
    t,
    μ,
    Σ,
    G = missing,
    b = missing,
    C = missing,
)
    push!(sol.t, t)
    push!(sol.μ, copy(μ))
    push!(sol.Σ, copy(Σ))
    push!(
        sol.backwards_transitions,
        (
            ismissing(G) ? missing : deepcopy(G),
            ismissing(b) ? missing : deepcopy(b),
            ismissing(C) ? missing : deepcopy(C),
        ),
    )

    return sol
end



function Base.getindex(sol::AbstractKalmanSolution, i::Int64)
    return (sol.μ[i], sol.Σ[i])
end
function Base.length(sol::AbstractKalmanSolution)
    return Base.length(sol.t)
end

function Base.lastindex(sol::AbstractKalmanSolution)
    return Base.lastindex(sol.t)
end

means(sol::AbstractKalmanSolution) = vecvec2mat(sol.μ)

stds(sol::AbstractKalmanSolution, mult_with::Float64 = 1.0) =
    vecvec2mat((map(M -> mult_with * sqrt.(LinearAlgebra.diag(M)), sol.Σ)))
