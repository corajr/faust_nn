# spectral_loss_fns.jl
export mag_spec, spectral_loss

import DSP
import FFTW

"""
    lags(x, n, t)

Samples signal `x` before sample `t` by 2^0, 2^1, ..., 2^(n-1).
Signal is considered 0 if we sample before it starts.
"""
function lags(x, n, t)
    x_sampled = zeros(n)
    for i = 0:n - 1
        t_lag = t - 2^i
        if t_lag > 0
            x_sampled[i + 1] = x[t_lag]
        end
    end
    return x_sampled
end

hann = DSP.Windows.hanning(256, zerophase=true)

"""
    mag_spec(x[, n][, hop_size][, window])

Computes the |STFT| of audio signal `x` using FFT size `n` with `window`.

If `n` is not given, it defaults to 256. `hop_size` is div(n, 2). 
`window` defaults to the Hann window.

Note that to let Zygote work we must avoid mutating arrays in this
function, so a window must always be supplied. The length of the
window overrides the `n` argument.

# Examples
```julia-repl
julia> x = rand(256, 1); Plots.heatmap(log1p.(mag_spec(x)'))
julia> mag_spec(x, DSP.Windows.hanning(512, zerophase=true))
```
"""
function mag_spec(x; n=256, hop_size=128, window=hann)
    if size(window, 1) != n
        n = size(window, 1)
    end
    hop_size = div(n, 2)
    n_complete_windows = div(size(x, 1), n)
    reduce(hcat, [
        abs.(FFTW.rfft(x[offset:offset + n - 1] .* window))
        for offset in 1:hop_size:((n_complete_windows * n) - hop_size) 
    ])
end

"""
    spectral_loss(x, x̂)

Computes the MSE difference between two signals in the frequency domain.
"""
function spectral_loss(x, x̂)
    Flux.mse(mag_spec(x), mag_spec(x̂))
end