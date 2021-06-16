import Faust
import FaustNN
import FiniteDifferences
import Flux
import FFTW
import IterTools
import PortAudio
import Zygote

block_size = 2048
samplerate = 16000

src = """
import("stdfaust.lib");

freq = 10 ^ hslider("freq", 0.0, 0.0, 1.0, 0.001);
Q = 10 ^ hslider("Q", 0.0, 0.0, 1.0, 0.001);
gain = hslider("gain", 0.0, 0.0, 1.0, 0.001);
process = fi.resonbp(freq, Q, gain);
"""
process = Faust.compile(src)

function set_params(d::Faust.DSPBlock, p::Vector{Float32})
    unsafe_store!(d.ui.paths["/score/freq"], p[1])
    unsafe_store!(d.ui.paths["/score/Q"], p[2])
    unsafe_store!(d.ui.paths["/score/gain"], p[3])
end

function f(x::Vector{Float32}, params::Vector{Float32})
    Zygote.ignore() do
        Faust.init(process, block_size=block_size, samplerate=samplerate)
        process.inputs = reshape(x, block_size, 1)
        set_params(process, params)
        vec(Faust.compute(process))
    end
end

function f(x::Matrix{Float32}, params::Vector{Float32})
    mapslices(v -> f(v, params), x, dims=[1])
end

struct FaustLayer
    params::Vector{Float32}
end

function FaustLayer(in::Integer; init = in -> rand(Float32, in))
    return FaustLayer(init(in))
end

function (m::FaustLayer)(x)
    f(x, m.params)
end

Flux.@functor FaustLayer

Zygote.@adjoint f(x, p) = f(x, p), ȳ -> (
    nothing,
    FiniteDifferences.j′vp(FiniteDifferences.central_fdm(5, 1), p -> f(x, p), ȳ, p)[1],
)
Zygote.refresh()

model = FaustLayer(3)
ps = Flux.params(model)
# loss(x, y) = FaustNN.harmonic_loss(model(x), y)
loss(x, y) = Flux.mse(model(x), y)

original_param = [0.5f0, 0.1f0, 0.5f0]
xs = 2 * rand(Float32, block_size, 50) .- 1
ys = mapslices(x -> f(x, original_param), xs, dims=[1])

opt = Flux.ADAM()
i = 0

c = Channel(0)
put!(c, model.params)

function display_loss(loss_total)
    global i
    Flux.@show((i, loss_total))
    put!(c, model.params)
end
throttled_cb = Flux.throttle(display_loss, 5)

function evalcb()
    global i
    loss_total = loss(xs, ys)
    throttled_cb(loss_total)
    i += 1
end

function play(c::Channel)
    d = Faust.compile(src)
    PortAudio.PortAudioStream(1, 1) do stream
        Faust.init(d, block_size=block_size, samplerate=Int(stream.samplerate))
        while true
            if isready(c)
                params = take!(c)
                set_params(d, params)
            end
            d.inputs = convert(Matrix{Float32}, read(stream, block_size))
            write(stream, Faust.compute(d))
        end
    end
end

t = @task play(c)

d_batch = Flux.Data.DataLoader((xs, ys), batchsize=50)
Flux.@epochs 20 Flux.train!(loss, ps, IterTools.ncycle(d_batch, 50), opt, cb = evalcb)