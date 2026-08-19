# Thin wrappers over scripts/. The scripts are the real interface — they work
# without make and are what CI calls — but `make test` is what people type.

.DEFAULT_GOAL := help
.PHONY: help setup build build-cuda test test-cpp test-python test-cuda \
        lint format bench profile docker clean

PYTHON ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create .venv and install everything this platform supports
	./scripts/setup.sh

build:  ## Build the portable C++ targets
	./scripts/build.sh

build-cuda:  ## Build with the CUDA targets (requires nvcc)
	./scripts/build.sh --cuda

test:  ## Run every suite this machine can run
	./scripts/test.sh

test-cpp: build  ## C++ tests only
	./build/tests/cpp/cudaforge_tests

test-python:  ## Python tests only
	$(PYTHON) -m pytest tests/python -q

test-cuda: build-cuda  ## CUDA tests (requires an NVIDIA GPU)
	./build-cuda/tests/cuda/cudaforge_cuda_tests

lint:  ## ruff, mypy, clang-format, CUDA and documentation checks
	./scripts/lint.sh

format:  ## Apply every formatter in place
	./scripts/lint.sh --fix

bench:  ## Run the benchmarks; results land in benchmarks/results
	./scripts/benchmark.sh

profile:  ## Nsight Systems and Compute profiles (requires an NVIDIA GPU)
	./scripts/profile.sh

docker:  ## Build the CUDA image
	docker build -t cudaforge:latest .

clean:  ## Remove build directories and caches
	rm -rf build build-* .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
