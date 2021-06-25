# FIXME: doesn't work.
# Enzyme.jl isn't able to handle vector-to-number functions yet.

import Enzyme
import Faust
import FFTW
import Statistics
using Test

function faustF(dsp::Faust.DSPBlock{T}, ŷ_desired::Array{T}) where T <: Faust.FaustFloat
    y = Faust.compute!(dsp)
    ŷ = FFTW.rfft(y)
    Statistics.mean((ŷ .- ŷ_desired) .^ 2)
end
function ∇faust(dsp::Faust.DSPBlock{T}, ddsp::Faust.DSPBlock{T}, ŷ_desired::Array{T}) where T <: Faust.FaustFloat
    Enzyme.gradient(faustF, Enzyme.Duplicated(dsp, ddsp), Enzyme.Const(ŷ_desired))
end

src = """
import("stdfaust.lib");

process = no.noise * hslider("gain", 1, 0, 1, 0.001);
"""
dsp = Faust.compile(src)
ddsp = Faust.compile(src)
ŷ_desired = zeros(Float32, 1025, 1)

# ∇faust(dsp, ddsp, ŷ_desired)
