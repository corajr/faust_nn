module FaustNN

import BSON
import CUDA
import Dates
import DSP
import FFTW
import FileIO
import Flux
import IterTools
# try
#     import LibSndFile
# catch e
#     # ignore double registration of OGG format.
# end

# include("harmonic_loss_fns.jl")

# n_fft = 256
# audio = FileIO.load("hum.wav")
# spec = mag_spec(audio, n_fft)

# ts = ((1:size(spec, 1)) * div(n_fft, 2)) / audio.samplerate
# n_fs = 12
# midikey2hz(mk) = 440.0 * 2^((mk-69.0)/12.0);
# fs = midikey2hz.(60:60+n_fs-1)
# oscillators = [sum(sin.(2*pi*fs*t)) for t in ts]

CUDA.allowscalar(false)

mutable struct Model
    m
    xs
    ys
    ps
    loss
end

function chord_model()
xs = reduce(hcat, [[0, 0,], [0, 1], [1, 0], [1, 1]]) |> Flux.gpu

#        0  1  2  3  4  5  6  7  8  9  t  e
major = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]

ys = reduce(hcat, [circshift(major, 7*i) for i in 0:3]) |> Flux.gpu

h = 12
m = Flux.Chain(
	Flux.Dense(2, 12, Flux.σ),
) |> Flux.gpu

loss(x, y) = Flux.mse(m(x), y)
ps = Flux.params(m)
Model(m, xs, ys, ps, loss)
end

function train_model(model)
opt = Flux.ADAM()

ts = () -> Dates.value(Dates.now()) - Dates.UNIXEPOCH

run_started = ts()
Base.Filesystem.mkpath("checkpoints")

checkpoint_period = 1000
i = 0

function display_loss()
    loss_total = model.loss(model.xs, model.ys)
    Flux.@show((i, loss_total))
end
throttled_cb = Flux.throttle(display_loss, 5)

function evalcb()
    throttled_cb()
    if i % checkpoint_period == 0
        m_cpu = Flux.cpu(model.m)
        BSON.@save "checkpoints/model-$run_started-$(lpad(i,3,'0')).bson" m_cpu opt loss_total
    end
    i += 1
end

d_batch = Flux.Data.DataLoader((model.xs, model.ys), batchsize = 4)
Flux.@epochs 10000 Flux.train!(model.loss, model.ps, d_batch, opt, cb = evalcb)
end

faust_nls = Dict(
    :tanh => "ma.tanh",
    :relu => "\\(x).(x * (x > 0))",
    :σ => "\\(x).(1.0 / (1.0 + exp(-x)))",
)

function gen_faust_code(model)
weight_buffer = IOBuffer()
bias_buffer = IOBuffer()
nls_buffer = IOBuffer()

m_cpu = Flux.cpu(model.m)
for (layer_idx, layer) in enumerate(m_cpu.layers)
    for (I, v) in pairs(IndexCartesian(), layer.W)
        println(weight_buffer, "w($(layer_idx - 1), $(I[2] - 1), $(I[1] - 1)) = $v;")
    end
    for (I, v) in pairs(IndexCartesian(), layer.b)
        println(bias_buffer, "b($(layer_idx - 1), $(I[1] - 1)) = $v;")
    end
    println(nls_buffer, "nl($(layer_idx-1)) = $(faust_nls[Symbol(layer.σ)]);\n")
end

for buf in [weight_buffer, bias_buffer, nls_buffer]
    seekstart(buf)
    println(countlines(buf))
end

weights = String(take!(weight_buffer));
biases = String(take!(bias_buffer));
layer_nls = String(take!(nls_buffer));

layer_sizes = vcat([size(layer.W, 2) for layer in m_cpu.layers], [size(last(m_cpu.layers).W, 1)])
layer_sizes_str = join(layer_sizes, ", ")

nn = """
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
""";

faust_code = """
import("stdfaust.lib");

$nn

notes = par(i, 12, _ * (1/12) * os.osc(ba.midikey2hz(60 + i))) :> _;
process = checkbox("2"), checkbox("1") : nn : notes <: _, _;
""";
faust_code
end

function compile_faust(faust_code, program_name = "faust_nn")
dsp_fname = "$program_name.dsp"
io = open(dsp_fname, "w")
write(io, faust_code)
close(io)

dsp_path = Base.Filesystem.abspath(dsp_fname)
faust_path = "C:\\Program Files\\Faust\\bin"
cd(() -> (
    # println(readdir("."));
    run(`bash.exe -c faust2sndfile $dsp_path`);
), faust_path)   

run(`bash.exe -c "/mnt/c/Users/coraj/Documents/faust_nn/$program_name hum_pred.wav"`, wait=true)

end
end
