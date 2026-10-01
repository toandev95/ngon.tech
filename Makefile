.PHONY: run build build-windows fmt lint

run:
	go run ./cmd/gateway -config config.toml

build:
	go build -o bin/gateway ./cmd/gateway

build-windows:
	go build -trimpath -ldflags="-s -w" -o ngon-gateway.exe ./cmd/gateway

fmt:
	gofmt -w cmd internal

lint:
	go vet ./...
