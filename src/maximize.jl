struct TraceObjective{F1, F2} <: Function
    f::F1
    ∇f::F2
    xtrace::Vector{Any}
    ftrace::Vector{Any}
    gtrace::Vector{Any}
end
TraceObjective(f, ∇f) = TraceObjective(f, ∇f, Any[], Any[], Any[])

(to::TraceObjective)(x) = to.f(x)
function ChainRulesCore.rrule(f::TraceObjective, x)
    v, g = f.f(x), f.∇f(x)
    push!(f.xtrace, copy(x))
    push!(f.ftrace, copy(v))
    push!(f.gtrace, copy(g))
    return v, Δ -> (NoTangent(), Δ * g)
end

function maximize_with_trace(f, ∇f, x₀, optimizer::IpoptAlg; power = 2.0, shift = 0.0, deflation_ub = 1000, deflation_radius = 0.0, prev_solutions = Any[], reduce = +, kwargs...)
    _of = TraceObjective(f, ∇f)
    of = x -> -_of(x[1:end-1])
    options = IpoptOptions(; kwargs...)
    model = Nonconvex.Model(of)
    N = length(x₀)
    Nonconvex.addvar!(model, fill(-10000, N), fill(10000, N))
    Nonconvex.addvar!(model, [0.0], [deflation_ub])
    function deflation(x, y)
        if length(prev_solutions) > 0
            d = zero(eltype(x))
            for sol in prev_solutions
                d = reduce(d, (1/max(0, norm(x - sol) - deflation_radius))^power + shift)
            end
            return d - y
        else
            return return -one(eltype(x))
        end
    end
    Nonconvex.add_ineq_constraint!(model, x -> deflation(x[1:end-1], x[end]))
    res = Nonconvex.optimize(model, optimizer, [x₀; deflation_ub / 2]; options)
    push!(prev_solutions, res.minimizer[1:end-1])
    return identity.(_of.xtrace), identity.(_of.ftrace), identity.(_of.gtrace)
end

function maximize_with_trace(f, ∇f, x₀, optimizer::Optim.AbstractOptimizer; prev_solutions = [], kwargs...)
    negf(x) = -f(x)
    g!(y, x) = (y .= .-∇f(x))

    function callback(states)
        # terminate if optimization encounters NaNs
        s = states[end]
        md = s.metadata
        return isnan(s.value) || any(isnan, md["x"]) || any(isnan, md["g(x)"])
    end
    options = Optim.Options(;
        store_trace=true, extended_trace=true, callback=callback, kwargs...
    )
    res = Optim.optimize(negf, g!, x₀, optimizer, options)

    xs = Optim.x_trace(res)::Vector{typeof(Optim.minimizer(res))}
    fxs = -Optim.f_trace(res)
    ∇fxs = map(tr -> -tr.metadata["g(x)"], Optim.trace(res))::typeof(xs)

    return xs, fxs, ∇fxs
end
