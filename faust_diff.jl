import Faust
import FaustNN
import FiniteDifferences
import Flux
import FFTW
import Zygote

block_size = 1024
samplerate = 16000

osc = Faust.compile("""
import("stdfaust.lib");
freq = 10 ^ hslider("log10_freq", 2.0, 2.0, 3.0, 0.001);
gain = hslider("gain", 0.0, 0.0, 1.0, 0.001);
process = os.oscs(freq) * gain;
""")

function oracle(ps)
    Faust.init(osc, block_size=block_size, samplerate=samplerate)
    unsafe_store!(osc.ui.paths["/score/log10_freq"], ps[1])
    unsafe_store!(osc.ui.paths["/score/gain"], ps[2])
    vec(Faust.compute(osc))
end

process = Faust.compile("""
import("stdfaust.lib");

freq = 10 ^ hslider("log10_freq", 2.0, 2.0, 3.0, 0.001);
Q = 200.0;
gain = hslider("gain", 0.0, 0.0, 1.0, 0.001);
process = no.noise : fi.resonbp(freq, Q, gain);
""")

function f(ps::Vector{Float32})
    Zygote.ignore() do
        Faust.init(process, block_size=block_size, samplerate=samplerate)
        unsafe_store!(process.ui.paths["/score/log10_freq"], ps[1])
        unsafe_store!(process.ui.paths["/score/gain"], ps[2])
        vec(Faust.compute(process))
    end
end

function f(ps::Matrix{Float32})
    mapslices(f, ps, dims = [1])
end

∇f(p) = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(5, 1), f, p)

Zygote.@adjoint f(p) = f(p), ȳ -> (ȳ' * ∇f(p)[1],)
Zygote.refresh()

model = f
ps = Flux.params(model)

xs = vec([collect(p) for p in Iterators.product(0:0.1f0:1, 0:0.1f0:1)])
ys = oracle.(xs)
xs = reduce(hcat, xs)
ys = reduce(hcat, ys)

loss(x, y) = harmonic_loss(f(x), y)

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
Flux.@epochs 10 Flux.train!(loss, ps, d_batch, opt, cb = evalcb)