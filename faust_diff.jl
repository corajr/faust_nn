import Faust
import FaustNN
import FiniteDifferences
import Flux
import FFTW
import Zygote

block_size = 2048
samplerate = 16000

process = Faust.compile("""
import("stdfaust.lib");

freq = 10 ^ hslider("log10_freq", 2.0, 2.0, 3.0, 0.001);
Q = 200.0;
gain = hslider("gain", 0.0, 0.0, 1.0, 0.001);
process = fi.resonbp(freq, Q, gain);
""")

function f(x::Vector{Float32}, params::Vector{Float32})
    Zygote.ignore() do
        Faust.init(process, block_size=block_size, samplerate=samplerate)
        process.inputs = reshape(x, block_size, 1)
        unsafe_store!(process.ui.paths["/score/log10_freq"], params[1])
        unsafe_store!(process.ui.paths["/score/gain"], params[2])
        vec(Faust.compute(process))
    end
end

function f(x::Matrix{Float32}, params::Vector{Float32})
    mapslices(v -> f(v, params), x, dims=[1])
end

struct FaustLayer
    params::Vector{Float32}
end

function FaustLayer(in::Integer; init = in -> ones(Float32, in))
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

model = FaustLayer(2)
ps = Flux.params(model)
loss(x, y) = FaustNN.harmonic_loss(model(x), y)

original_param = [0.5f0, 0.5f0]
xs = 2 * rand(Float32, block_size, 50) .- 1
ys = mapslices(x -> f(x, original_param), xs, dims=[1])

opt = Flux.Descent()
i = 0

function display_loss(loss_total)
    global i
    Flux.@show((i, loss_total))
end
throttled_cb = Flux.throttle(display_loss, 5)

function evalcb()
    global i
    loss_total = loss(xs, ys)
    throttled_cb(loss_total)
    i += 1
end

d_batch = Flux.Data.DataLoader((xs, ys))
Flux.@epochs 20 Flux.train!(loss, ps, d_batch, opt, cb = evalcb)