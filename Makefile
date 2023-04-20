image ?= parkeraddison/policyrecon
devimage ?= parkeraddison/policyrecon-devcontainer
tag ?= latest

.PHONY: help # Show this help message. Default when no target provided.
help:
	@echo "Available Targets\n-----------------"
	@grep '^.PHONY: .* #' Makefile | sed 's,\.PHONY: \(.*\) # \(.*\),  \1#\2\n,' | column -ts "#"

.PHONY: build # Build the base container image. Available params: image, tag
build:
	docker build -t ${image}:${tag} .

.PHONY: build-dev # Build the devcontainer image. Available params: devimage, tag
build-dev:
	docker build -t ${devimage}:${tag} -f .devcontainer/Dockerfile .

.PHONY: launch # Launch a base container image for easy manual pipeline runs. Available params: image, tag
launch:
	docker run -it  \
		-v $$(pwd):/workspace -w /workspace \
		-v $$(pwd)/.docker-cache:/root \
		${image}:${tag} \
		bash

.PHONY: run # Run pipeline from a base container image. Available params: image, tag
run:
	docker run -it  \
		-v $$(pwd):/workspace -w /workspace \
		-v $$(pwd)/.docker-cache:/root \
		${image}:${tag} \
		python pipeline.py
