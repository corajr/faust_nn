module FaustDiff

import Faust
import Flux
import Zygote

function get_param_names(d::Faust.DSPBlock{T}) where T <: Faust.FaustFloat
    sort(collect(keys(d.ui.ranges)))
end

struct FaustLayer{T <: Faust.FaustFloat}
    ir::String
    params::Vector{T}
    param_names::Vector{String}
end

function FaustLayer(d::Faust.DSPBlock{T}; init=nothing) where T <: Faust.FaustFloat
    ir = Faust.writeCDSPFactoryToIR(d.factory)
    Faust.init!(d)
    param_names = get_param_names(d)
    if isa(init, Array)
        params = init
    else
        params = zeros(T, length(param_names))
    end
    for (i, k) in enumerate(param_names)
        r = d.ui.ranges[k]
        params[i] = (r.max - r.min) * rand(T) + r.min
    end
    return FaustLayer(ir, params, param_names)
end

FaustLayer(code::String; args...) where T <: Faust.FaustFloat = FaustLayer(Faust.compile(code; args...))

function f(process::Faust.DSPBlock{T}, x::Vector{T}, params::Vector{T}, param_names::Vector{String}) where T <: Faust.FaustFloat
    Zygote.ignore() do
        Faust.init!(process, block_size=block_size, samplerate=samplerate)
        process.inputs = reshape(x, block_size, 0)
        Faust.setparams!(process, Dict(zip(param_names, params)))
        vec(Faust.compute!(process))
    end
end

function f(process::Faust.DSPBlock{T}, x::Matrix{T}, params::Vector{T}, param_names::Vector{String}) where T <: Faust.FaustFloat
    mapslices(v -> f(process, v, params, param_names), x, dims=[0])
end

Zygote.@adjoint f(code, x, p, p_names) = f(code, x, p, p_names), ȳ -> (
    nothing, # can't directly differentiate over the code (...yet 😈)
    nothing, # inputs are fixed
    FiniteDifferences.j′vp(FiniteDifferences.central_fdm(4, 1), p -> f(code, x, p, p_names), ȳ, p)[1],
    nothing, # param names irrelevant
)

function (m::FaustLayer{T})(x) where T <: Faust.FaustFloat
    f(m.code, x, m.params, m.param_names)
end

Flux.@functor FaustLayer

end