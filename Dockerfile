FROM nvidia/cuda:11.3.0-devel-ubuntu20.04

ENV JULIA_URL https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.1-linux-x86_64.tar.gz
ENV JULIA_PATH /usr/local/julia
ENV PATH $JULIA_PATH/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl faust libsndfile-dev

RUN set -eux; \
    \
    curl -fL -o julia.tar.gz "$JULIA_URL"; \
    mkdir -p "$JULIA_PATH"; \
    tar -xzf julia.tar.gz -C "$JULIA_PATH" --strip-components 1; \
    rm julia.tar.gz; \
    julia --version
