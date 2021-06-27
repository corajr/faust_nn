using FaustNN
import Flux
using Test
using Zygote

@testset "chord_model" begin
    model = chord_model()
    @test size(model.xs, 2) == size(model.ys, 2)
    @test model.loss(model.xs, model.ys) > 0.0
end

@testset "FaustLayer" begin
    p = FaustLayer("""process = _ * hslider("gain", 0.0, 0.0, 1.0, 0.001);""")
    @test length(p.params) == 1
    @test p.param_names == ["/score/gain"]

    loss(x, y) = Flux.mse(p(x), y)
    ps = Flux.params(p)
    x = rand(FaustNN.getfloattype(p.process), p.process.block_size, 1) 
    grads = Flux.gradient(ps) do
        loss(x, zeros(FaustNN.getfloattype(p.process), size(x)))
    end
    ∂ps = grads[ps[1]]
    @test ∂ps[1] > 0
end