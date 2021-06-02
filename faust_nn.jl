import Pkg; Pkg.add(["CUDA", "DSP", "FFTW", "FileIO", "Flux", "IterTools", "LibSndFile"])
import CUDA
import DSP
import FFTW
import FileIO
import Flux
import IterTools
import LibSndFile

audio = FileIO.load("hum.wav")

sample_len = 512
n_clips = size(audio, 1) ÷ sample_len
ts = 1:sample_len:sample_len * n_clips
xs = reduce(vcat, [audio[t:t+sample_len] for t in ts]) |> Flux.gpu
ys = xs

enc = Flux.Chain(
    Flux.Dense(sample_len, sample_len >> 1, tanh),
    Flux.Dense(sample_len >> 1, sample_len >> 2, tanh),
)
dec = Flux.Chain(
    Flux.Dense(sample_len >> 2, sample_len >> 1, tanh),
    Flux.Dense(sample_len >> 1, sample_len, tanh),
)

model = Flux.Chain(enc, dec) |> Flux.gpu

opt = Flux.ADAM()

loss(x, y) = Flux.mse(vec(m(x)), y)
ps = Flux.params(m)

i = 0
function evalcb()
    y_pred = vec(m(xs))
    loss_total = Flux.mse(y_pred, xs)
    if i % 1 == 0
        Flux.@show((i, loss_total))
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
process = no.multinoise($(first(layer_sizes))) : nn;
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
