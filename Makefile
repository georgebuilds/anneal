.PHONY: build test fmt vet lint coverage coverage-html

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

# -covermode=atomic is required when packages are tested in parallel (default for ./...).
# tail -1 prints the project-wide total, which CI keys on for the visible percentage.
coverage:
	go test ./... -coverprofile=coverage.out -covermode=atomic
	go tool cover -func=coverage.out | tail -1

coverage-html: coverage
	go tool cover -html=coverage.out -o coverage.html
