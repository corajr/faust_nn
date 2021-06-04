# harmonic_loss_fns.jl

"""
    lags(x, n, t)

Samples signal `x` before sample `t` by 2^0, 2^1, ..., 2^(n-1).
Signal is considered 0 if we sample before it starts.
"""
function lags(x, n, t)
    x_sampled = zeros(n)
    for i = 0:n-1
        t_lag = t - 2^i
        if t_lag > 0
            x_sampled[i + 1] = x[t_lag]
        end
    end
    return x_sampled
end

"""
    mag_spec(x[, n])

Computes the magnitude spectrum of audio signal `x` using FFT size `n`.

If `n` is not given, it defaults to 256.

# Examples
```julia-repl
julia> Plots.heatmap(log1p.(mag_spec(x)'))
```
"""
function mag_spec(x, n = 256)
    hann = DSP.Windows.hanning(n, zerophase=true)
    hop_size = div(n, 2)
    n_pos_freqs = div(n, 2) + 1
    n_hops = div(size(x, 1), hop_size)
    spec = zeros(n_hops, n_pos_freqs)
    for i=1:n_hops
        offset = ((i - 1) * hop_size) + 1
        windowed = x[offset:offset+n-1] .* hann
        spec[i, :] = abs.(FFTW.rfft(windowed))
    end
    spec
end

"""
    harmonic_loss(x, x̂)

Computes the MSE difference between two signals.
"""
function harmonic_loss(x, x̂)
    Flux.mse(mag_spec(x), mag_spec(x̂))
end