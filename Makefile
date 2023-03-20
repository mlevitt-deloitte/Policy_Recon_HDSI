image=parkeraddison/policyrecon-devcontainer
tag=latest

@PHONY: all
all: build push

@PHONY: build
build:
	docker build -t ${image}:${tag} .devcontainer

@PHONY: push
push:
	docker push ${image}:${tag}
