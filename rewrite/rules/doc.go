// Package rules contains the compiled rewrite rule sets: symbolic arithmetic, gradient, and scheduler rules.
//
//go:generate go run ../gen/main.go -pkg rules -in symbolic.upat -out symbolic_gen.go
package rules
