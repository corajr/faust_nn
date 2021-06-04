# faust-nn

Some experiments using NNs in Faust.

Basic set-up inspired by JOS's example:
https://faustdoc.grame.fr/examples/filtering/#dnn


```shell
docker build . --tag faust_nn
docker run -it --rm --gpus all \
    --mount type=bind,source="$(pwd)",target=/data \
   faust_nn
```
