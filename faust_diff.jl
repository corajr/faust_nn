import Distributed
import Faust
import FaustNN
import FiniteDifferences
import Flux
import FFTW
import IterTools
import PortAudio
import Zygote

# create an additional process to play sound through.
Distributed.addprocs(1);
wid = Distributed.workers()[1];
c = Distributed.RemoteChannel(()->Channel(1));

module_dir = pwd()
Distributed.@everywhere import Faust, PortAudio
Distributed.@everywhere block_size=2048
Distributed.@everywhere samplerate=44100
Distributed.@everywhere function play(c; code="process = 0", block_size=1024)
    d = Faust.compile(code)
    Faust.init!(d)

    # require a device matching the input and output size.
    devices = PortAudio.devices()
    dev = filter(x -> x.maxinchans >= size(d.inputs, 2) && x.maxoutchans >= size(d.outputs, 2), devices)[1]
    println("Device: $dev")

    PortAudio.PortAudioStream(in, out) do stream
        println("Sample rate: $(stream.samplerate)")
        Faust.init!(d, block_size=block_size, samplerate=Int(stream.samplerate))
        while true
            if isready(c)
                model = take!(c)
                Faust.setparams!(d, Dict(zip(model.param_names, model.params)))
            end
            d.inputs = read(stream, block_size)
            write(stream, Faust.compute!(d))
        end
    end
end


params_src = """
import("stdfaust.lib");

freq = 10 ^ hslider("log10_freq", log10(440), log10(20), log10(20000), 0.001);
Q = 10 ^ hslider("log10_Q", 1, 1, 3, 0.001);
gain = hslider("gain", 0.0, 0.0, 1.0, 0.001);
"""

src = string(params_src, """
process = fi.resonbp(freq, Q, gain);
""")

model = FaustNN.FaustLayer(src)
ground_truth_model = FaustNN.FaustLayer(src; init=[log10(261.625565f0), 3.0f0, 0.4f0])

ps = Flux.params(model)
loss(x, y) = FaustNN.spectral_loss(model(x), y)

opt = Flux.ADAM()
i = 0

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


noise_src = string(params_src, """
process = no.noise : fi.resonbp(freq, Q, gain);
""")

# TODO: PortAudio doesn't seem to work with Distributed.
# Try spawning a separate process and controlling with OSC.
# f = Distributed.@spawnat wid play(c; code=noise_src)

t = @async play(c; code=noise_src)

xs = 2 * rand(Float32, block_size, 50) .- 1
ys = ground_truth_model(xs)
d_batch = Flux.Data.DataLoader((xs, ys), batchsize=50)
Flux.@epochs 20 Flux.train!(loss, ps, IterTools.ncycle(d_batch, 50), opt, cb = evalcb)