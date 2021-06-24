export Model, chord_model, train_model, gen_faust_code

import BSON
import CUDA
import Dates
import DSP
import Faust
import FFTW
import FileIO
import Flux
import IterTools
import LinearAlgebra
# try
#     import LibSndFile
# catch e
#     # ignore double registration of OGG format.
# end

include("spectral_loss_fns.jl")

CUDA.allowscalar(false)

mutable struct Model
    m
    xs
    ys
    ps
    loss
    opt
end

function chord_model(hidden_layer_size=1)
xs = hcat(
    reduce(hcat, [[i / 12, 0] for i in 0:11]),
    reduce(hcat, [[i / 12, 1] for i in 0:11]),
 ) |> Flux.gpu

#                                 0  1  2  3  4  5  6  7  8  9  t  e
major = convert(Vector{Float32}, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
minor = convert(Vector{Float32}, [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])

ys = hcat(
    reduce(hcat, [circshift(major, 7*i) for i in 0:11]),
    reduce(hcat, [circshift(minor, 7*i) for i in 0:11]),
) |> Flux.gpu

layer_sizes = 2, hidden_layer_size, 12
layers = collect(zip(layer_sizes, Iterators.drop(layer_sizes, 1)))
activations = [Flux.tanh, Flux.σ]
m = Flux.Chain([
    Flux.Dense(i, o, σ)
    for ((i, o), σ) in zip(layers, activations)
]...)|> Flux.gpu

loss(x, y) = Flux.mse(m(x), y)
ps = Flux.params(m)
opt = Flux.ADAM()
Model(m, xs, ys, ps, loss, opt)
end

function train_model(model)
ts = () -> Dates.value(Dates.now()) - Dates.UNIXEPOCH

run_started = ts()
Base.Filesystem.mkpath("checkpoints")

checkpoint_period = 1000
i = 0

function display_loss(loss_total)
    Flux.@show((i, loss_total))
end
throttled_cb = Flux.throttle(display_loss, 5)

function evalcb()
    loss_total = model.loss(model.xs, model.ys)
    throttled_cb(loss_total)
    if i % checkpoint_period == 0
        m_cpu = Flux.cpu(model.m)
        BSON.@save "checkpoints/model-$run_started-$(lpad(i,3,'0')).bson" m_cpu opt = model.opt loss_total
    end
    loss_total < 1f-5 && Flux.stop()
    i += 1
end

d_batch = Flux.Data.DataLoader((model.xs, model.ys), shuffle = true)
Flux.@epochs 1000 Flux.train!(model.loss, model.ps, IterTools.ncycle(d_batch, 100), model.opt, cb = evalcb)
println("loss: $(model.loss(model.xs, model.ys))")
end

faust_nls = Dict(
    :tanh => "ma.tanh",
    :sin => "sin",
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

# input_type = "checkbox(\"%2i\")"
input_type = "hslider(\"%2i\", 0, 0, 1, 1/12)"
input = "par(i, $(layer_sizes[1]), $input_type)"

faust_code = """
import("stdfaust.lib");

$nn

notes = par(i, 12, _ * (1/12) * os.osc(ba.midikey2hz(60 + i))) :> _;
process = $input : nn : notes <: _, _;
""";
faust_code
end