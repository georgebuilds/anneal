.PHONY: build test fmt vet lint

BINARY := bin/anneal

build:
	mkdir -p bin
	go build -o $(BINARY) ./cmd/anneal

test:
	go test ./...

fmt:
	gofmt -w .

vet:
	go vet ./...

lint:
	golangci-lint run ./...
