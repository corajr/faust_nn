using FaustNN
using Test

@testset "chord_model" begin
    model = chord_model()
    @test size(model.xs, 2) == size(model.ys, 2)
    @test model.loss(model.xs, model.ys) > 0.0
end

@testset "FaustLayer" begin
    p = FaustLayer("""process = hslider("out", 0.0, 0.0, 1.0, 0.001);""")
    @test length(p.params) == 1
    @test p.params[1] >= 0.0 && p.params[1] <= 1.0
    @test length(p.param_names) == 1
end