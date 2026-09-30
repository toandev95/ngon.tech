.PHONY: run build fmt lint

run:
	go run ./cmd/gateway

build:
	go build -o bin/gateway ./cmd/gateway

fmt:
	gofmt -w cmd internal

lint:
	go vet ./...
