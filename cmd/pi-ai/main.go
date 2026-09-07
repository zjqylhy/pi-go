// Command pi-ai is a minimal demo CLI over the pi-go ai package. It can list
// built-in providers/models and stream a completion.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/zjqylhy/pi-go/ai"
	_ "github.com/zjqylhy/pi-go/llmlog"
)

func main() {
	provider := flag.String("provider", "", "provider id")
	model := flag.String("model", "", "model id")
	list := flag.Bool("models", false, "list providers/models and exit")
	flag.Parse()

	m := ai.BuiltinModels()

	if *list || (*provider == "" && *model == "") {
		listModels(m)
		return
	}

	mdl := m.GetModel(*provider, *model)
	if mdl == nil {
		fmt.Fprintf(os.Stderr, "unknown model %s/%s\n", *provider, *model)
		os.Exit(1)
	}

	prompt := strings.Join(flag.Args(), " ")
	if prompt == "" {
		fmt.Fprintln(os.Stderr, "provide a prompt as arguments")
		os.Exit(1)
	}

	run(m, mdl, prompt)
}

func listModels(m ai.MutableModels) {
	for _, p := range m.GetProviders() {
		fmt.Printf("%s (%s)\n", p.Name(), p.ID())
		for _, mdl := range p.GetModels() {
			fmt.Printf("  %s\n", mdl.ID)
		}
	}
}

func run(m ai.MutableModels, mdl *ai.Model, prompt string) {
	ctxt := &ai.Context{
		Messages: []ai.Message{
			&ai.UserMessage{Role: "user", Content: []ai.ContentBlock{ai.TextBlock(prompt)}},
		},
	}
	opts := &ai.SimpleStreamOptions{}
	opts.Ctx = context.Background()

	stream := m.StreamSimple(mdl, ctxt, opts)
	for e := range stream.Events() {
		if e.Type == ai.EventTextDelta {
			fmt.Print(e.Delta)
		}
	}
	fmt.Println()

	result := stream.Result()
	if result.StopReason == ai.StopError {
		fmt.Fprintf(os.Stderr, "error: %s\n", result.ErrorMessage)
		os.Exit(1)
	}
	fmt.Printf("\n— %s · %d input / %d output tokens\n", result.StopReason, result.Usage.Input, result.Usage.Output)
}
