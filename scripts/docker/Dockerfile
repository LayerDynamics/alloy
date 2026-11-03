# syntax=docker/dockerfile:1

# scripts/docker/Dockerfile
#
# Linux arm64 builder for:
#   - bindgen_output_stable
#   - vendored_build_reproducible
#
# Build and run with --platform linux/arm64.
#
# Optional base-image override:
#   --build-arg RUST_IMAGE=rust:1.99.1-bookworm@sha256:<digest>
#
# libclang is version-pinned because its version affects bindgen output.

ARG RUST_IMAGE=rust:1.99.1-bookworm
FROM ${RUST_IMAGE}

ARG LLVM_VERSION=14
ARG LIBCLANG_PKG_VERSION=1:14.0.6-12

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      cmake \
      libblas-dev \
      liblapack-dev \
      liblapacke-dev \
      libclang1-${LLVM_VERSION}=${LIBCLANG_PKG_VERSION} \
      libclang-common-${LLVM_VERSION}-dev=${LIBCLANG_PKG_VERSION} \
      libclang-${LLVM_VERSION}-dev=${LIBCLANG_PKG_VERSION} \
 && rm -rf /var/lib/apt/lists/*

RUN rustup component add rustfmt clippy

ENV LIBCLANG_PATH=/usr/lib/llvm-${LLVM_VERSION}/lib
ENV CARGO_TERM_COLOR=never

WORKDIR /work