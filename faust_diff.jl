import Faust
import FaustNN
import FiniteDifferences
import Flux
import FFTW
import IterTools
import PortAudio
import Zygote

block_size = 2048
samplerate = 44100

params_src = """
import("stdfaust.lib");

freq = 10 ^ hslider("log10_freq", log10(110), log10(110), log10(880), 0.001);
Q = 10 ^ hslider("log10_Q", 1, 1, 3, 0.001);
gain = hslider("gain", 0.0, 0.0, 1.0, 0.001);
"""

src = string(params_src, """
process = fi.resonbp(freq, Q, gain);
""")
stereo_src = string(params_src, """
process = sp.stereoize(fi.resonbp(freq, Q, gain));
""")
process = Faust.compile(src)

function get_param_names(d::Faust.DSPBlock)
    sort(collect(keys(d.ui.ranges)))
end

function f(x::Vector{T}, params::Vector{T}, param_names::Vector{String}) where T <: Faust.FaustFloat
    Zygote.ignore() do
        Faust.init!(process, block_size=block_size, samplerate=samplerate)
        process.inputs = reshape(x, block_size, 1)
        Faust.setparams!(process, Dict(zip(param_names, params)))
        vec(Faust.compute!(process))
    end
end

function f(x::Matrix{T}, params::Vector{T}, param_names::Vector{String}) where T <: Faust.FaustFloat
    mapslices(v -> f(v, params, param_names), x, dims=[1])
end

struct FaustLayer{T <: Faust.FaustFloat}
    params::Vector{T}
    param_names::Vector{String}
end

function FaustLayer(d::Faust.DSPBlock{T}) where T <: Faust.FaustFloat
    Faust.init!(d)
    param_names = get_param_names(d)
    params = zeros(T, length(param_names))
    for (i, k) in enumerate(param_names)
        r = d.ui.ranges[k]
        params[i] = (r.max - r.min) * rand(T) + r.min
    end
    return FaustLayer(params, param_names)
end

function (m::FaustLayer)(x)
    f(x, m.params, m.param_names)
end

Flux.@functor FaustLayer

Zygote.@adjoint f(x, p, names) = f(x, p, names), ȳ -> (
    nothing,
    FiniteDifferences.j′vp(FiniteDifferences.central_fdm(5, 1), p -> f(x, p, names), ȳ, p)[1],
    nothing,
)
Zygote.refresh()

model = FaustLayer(Faust.compile(src))
ps = Flux.params(model)
loss(x, y) = FaustNN.spectral_loss(model(x), y)

opt = Flux.ADAM()
i = 0

c = Channel(1)

function display_loss(loss_total)
    Flux.@show((i, loss_total))
    put!(c, model)
end
throttled_cb = Flux.throttle(display_loss, 5)

function evalcb()
    global i
    loss_total = loss(xs, ys)
    throttled_cb(loss_total)
    i += 1
end


# TODO: try spawning a separate process and controlling with OSC.
function play(c; in=2, out=2)
    d = Faust.compile(stereo_src)
    PortAudio.PortAudioStream(in, out) do stream
        if d.dsp == C_NULL
            println(stream.samplerate)
            Faust.init!(d, block_size=block_size, samplerate=Int(stream.samplerate))
        end
        block = 0
        while true
            if block % 16 == 0 && isready(c)
                model = take!(c)
                Faust.setparams!(d, Dict(zip(model.param_names, model.params)))
            end
            d.inputs = convert(Matrix{Float32}, read(stream, block_size))
            write(stream, Faust.compute!(d))
            block = block + 1
        end
    end
end

devices = PortAudio.devices()
println(devices)
dev = filter(x -> x.maxinchans == 2 && x.maxoutchans == 2, devices)[1]

t = Threads.@spawn play(c; in=dev, out=dev)

original_param = [log10(440f0), 2f0, 0.5f0]
xs = 2 * rand(Float32, block_size, 50) .- 1
ys = f(xs, original_param, model.param_names)
d_batch = Flux.Data.DataLoader((xs, ys), batchsize=50)

Flux.@epochs 20 Flux.train!(loss, ps, IterTools.ncycle(d_batch, 50), opt, cb = evalcb)
