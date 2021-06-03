import Pkg; Pkg.add(["BSON", "CUDA", "DSP", "FFTW", "FileIO", "Flux", "IterTools", "LibSndFile"])
import BSON
import CUDA
import Dates
import DSP
import FFTW
import FileIO
import Flux
import IterTools
import LibSndFile

audio = FileIO.load("hum.wav")

samples = size(audio, 1)
dur = samples / audio.samplerate

ts = (1:samples) / samples
n_fs = 256
fs = 1:n_fs
augment(t) = vcat([t], sin.(2*pi*fs*t))
xs = reduce(hcat, augment.(ts)) |> Flux.gpu
ys = convert(Vector{Float32}, audio[:, 1]) |> Flux.gpu

n = 1 + n_fs
h = n_fs * 2

m = Flux.Chain(
    Flux.Dense(n, h, tanh),
    Flux.Dense(h, 1, tanh),
) |> Flux.gpu

opt = Flux.ADAM()

loss(x, y) = Flux.mse(m(x), y)
ps = Flux.params(m)

test_batch = Flux.Data.DataLoader((xs, ys), batchsize=256)

i = 0
function evalcb()
    loss_total = 0.0
    for d in test_batch
        loss_total += loss(d...)
    end
    Flux.@show((i, loss_total))
    if i % 10 == 0
        ts = Dates.value(Dates.now()) - Dates.UNIXEPOCH
        m_cpu = Flux.cpu(m)
        BSON.@save "model-$ts.bson" m_cpu opt
    end
    global i += 1
end
throttled_cb = Flux.throttle(evalcb, 5)

d_batch = Flux.Data.DataLoader((xs, ys), batchsize=256, shuffle=true)
Flux.@epochs 10 Flux.train!(loss, ps, d_batch, opt, cb = throttled_cb)

weight_buffer = IOBuffer()
bias_buffer = IOBuffer()

for (layer_idx, layer) in enumerate(m.layers)
    for (I, v) in pairs(IndexCartesian(), layer.W)
        println(weight_buffer, "w($(layer_idx - 1), $(I[2] - 1), $(I[1] - 1)) = $v;")
    end
    for (I, v) in pairs(IndexCartesian(), layer.b)
        println(bias_buffer, "b($(layer_idx - 1), $(I[1] - 1)) = $v;")
    end
end

weights = String(take!(weight_buffer))
biases = String(take!(bias_buffer))
layer_sizes = vcat([size(layer.W, 2) for layer in m.layers], [size(last(m.layers).W, 1)])
layer_sizes_str = join(layer_sizes, ", ")
faust_nls = Dict(
    :tanh => "ma.tanh",
    :relu => "\\(x).(x * (x > 0))"
    :σ => "\\(x).(1.0 / (1.0 + ma.exp(-x)))",
)
layer_nls = join(["  nl($(i-1)) = $(faust_nls[Symbol(layer.σ)]);\n" for (i, layer) in enumerate(dec.layers)])

faust_nn = """
layerSizes = $layer_sizes_str;

// w(layer, node_from, node_to)
$weights
$biases

M = ba.count(layerSizes);
N(l) = ba.take(l+1, layerSizes); // Nodes per layer

nn = seq(m, M-1, layer(m))
with {
  layer(m) = weights(m) :> biases(m) : nonlinearities(m);
  weights(m) = par(n, N(m), _ <: wts(m, n));
  wts(m, n) = par(k, N(m+1), *(w(m, n, k)));
  biases(m) = par(n, N(m+1), +(b(m, n)));
  nonlinearities(m) = par(n, N(m+1), nl(m));
$layer_nls
};
"""

faust_code = """
import("stdfaust.lib");

$faust_nn

augment(t) = t, par(f, $n_fs, sin(2*ma.PI*f*t));
process = os.lf_sawpos(1.0 / $dur) : augment : nn <: _, _;
"""

program_name = "faust_nn"
dsp_fname = "$program_name.dsp"
io = open(dsp_fname, "w")
write(io, faust_code)
close(io)

cmd = `faust2sndfile $dsp_fname`  
run(cmd)

gen_audio_cmd = `./$program_name hum.wav hum_pred.wav`
run(gen_audio_cmd)
