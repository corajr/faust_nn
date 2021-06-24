export FaustLayer

import Faust
import FiniteDifferences
import Flux
import Zygote

function get_param_names(d::Faust.DSPBlock{T}) where T <: Faust.FaustFloat
    sort(collect(keys(d.ui.ranges)))
end

struct FaustLayer{T <: Faust.FaustFloat}
    process::Faust.DSPBlock{T}
    params::Vector{T}
    param_names::Vector{String}
end

getfloattype(d::Faust.DSPBlock{T}) where {T} = T

function FaustLayer(code::String; init=nothing, samplerate=44100, args...)
    d = Faust.compile(code; args...)
    Faust.init!(d; samplerate=samplerate)
    param_names = get_param_names(d)
    if isa(init, Array)
        params = init
    else
        params = zeros(getfloattype(d), length(param_names))
    end
    for (i, k) in enumerate(param_names)
        r = d.ui.ranges[k]
        params[i] = (r.max - r.min) * rand(getfloattype(d)) + r.min
    end
    return FaustLayer(d, params, param_names)
end

function f(process::Faust.DSPBlock{T}, x::Vector{T}, params::Vector{T}, param_names::Vector{String}) where T <: Faust.FaustFloat
    Zygote.ignore() do
        block_size = size(x, 1)
        Faust.init!(process, block_size=block_size, samplerate=process.samplerate)
        process.inputs = reshape(x, block_size, 1)
        Faust.setparams!(process, Dict(zip(param_names, params)))
        vec(Faust.compute!(process))
    end
end

function f(process::Faust.DSPBlock{T}, x::Matrix{T}, params::Vector{T}, param_names::Vector{String}) where T <: Faust.FaustFloat
    mapslices(v -> f(process, v, params, param_names), x, dims=[1])
end

Zygote.@adjoint f(process, x, p, p_names) = f(process, x, p, p_names), ȳ -> (
    nothing, # can't directly differentiate over the code (...yet 😈)
    nothing, # inputs are fixed
    FiniteDifferences.j′vp(FiniteDifferences.central_fdm(5, 1), p -> f(process, x, p, p_names), ȳ, p)[1],
    nothing, # param names irrelevant
)

function (m::FaustLayer{T})(x) where T <: Faust.FaustFloat
    f(m.process, x, m.params, m.param_names)
end

Flux.@functor FaustLayer

function Base.show(io::IO, m::FaustLayer{T}) where T <: Faust.FaustFloat
    print(io, "Faust(")
    print(io, Dict(zip(m.param_names, m.params)))
    print(io, ")")
  end
