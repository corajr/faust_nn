import Faust
import FaustNN
import FiniteDifferences
import Flux
import FFTW
import Zygote

block_size = 2048
samplerate = 16000

osc = Faust.compile("""
import("stdfaust.lib");
freq = 10 ^ hslider("log10_freq", 2.0, 2.0, 3.0, 0.001);
gain = hslider("gain", 0.0, 0.0, 1.0, 0.001);
process = os.oscs(freq) * gain;
""")

function oracle(x)
    Faust.init(osc, block_size=block_size, samplerate=samplerate)
    unsafe_store!(osc.ui.paths["/score/log10_freq"], x[1])
    unsafe_store!(osc.ui.paths["/score/gain"], x[2])
    vec(Faust.compute(osc))
end

process = Faust.compile("""
import("stdfaust.lib");

freq = 10 ^ hslider("log10_freq", 2.0, 2.0, 3.0, 0.001);
Q = 200.0;
gain = hslider("gain", 0.0, 0.0, 1.0, 0.001);
process = no.noise : fi.resonbp(freq, Q, gain);
""")

function f(x::Vector{Float32})
    Zygote.ignore() do
        Faust.init(process, block_size=block_size, samplerate=samplerate)
        unsafe_store!(process.ui.paths["/score/log10_freq"], x[1])
        unsafe_store!(process.ui.paths["/score/gain"], x[2])
        vec(Faust.compute(process))
    end
end

function f(x::Matrix{Float32})
    mapslices(f, x, dims = [1])
end

Zygote.@adjoint f(x) = f(x), ȳ -> (FiniteDifferences.j'vp(FiniteDifferences.central_fdm(5, 1), f, ȳ, x),)
Zygote.refresh()

model = Flux.Chain(
    Flux.Dense(2, 2),
    f,
)
ps = Flux.params(model)
loss(x, y) = FaustNN.harmonic_loss(model(x), y)

xs = vec([collect(p) for p in Iterators.product(0.0f0:0.1f0:1f0, 0.5f0:0.1f0:1f0)])
ys = oracle.(xs)
xs = reduce(hcat, xs)
ys = reduce(hcat, ys)

opt = Flux.ADAM()
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