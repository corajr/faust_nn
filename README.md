# faust-nn

Some experiments using NNs in Faust.

## FaustLayer

Using finite difference methods, try to differentiate over the Faust program.

See [faust_diff.jl](faust_diff.jl).

## chord_model
Basic set-up inspired by JOS's example:
https://faustdoc.grame.fr/examples/filtering/#dnn

Run `julia mlp.jl` to train.

Takes two slider inputs:
[0-1] chord root
[0-1] chord quality (0 = major, 1 = minor)

Gives back the pitch classes of the chord.

```julia
julia> model = chord_model()
julia> train_model(model)
julia> faust_code = gen_faust_code(model)
```

## Usage notes

### Docker

Shouldn't be needed now that `faust_jll` is uploaded.

```shell
docker build . --tag faust_nn
docker run -it --rm --gpus all \
    --mount type=bind,source="$(pwd)",target=/data \
   faust_nn
```
