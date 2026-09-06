// Command pi-agent is an interactive coding-agent CLI that wires together the
// pi-go ai, agent, harness tools, session, and (optionally) tui packages.
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"pi-go/agent"
	"pi-go/ai"
	"pi-go/compaction"
	"pi-go/harness"
	"pi-go/session"
	"pi-go/tui"
)

func main() {
	provider := flag.String("provider", "anthropic", "provider id")
	modelID := flag.String("model", "claude-sonnet-4-5", "model id")
	dir := flag.String("dir", defaultSessionsDir(), "session storage directory")
	doNew := flag.Bool("new", false, "start a fresh session instead of resuming")
	doModels := flag.Bool("models", false, "list providers/models and exit")
	doTUI := flag.Bool("tui", false, "use the terminal UI (differential rendering)")
	flag.Parse()

	if *doModels {
		listModels()
		return
	}

	var err error
	if *doTUI {
		err = runTUI(*provider, *modelID, *dir, *doNew)
	} else {
		err = run(*provider, *modelID, *dir, *doNew)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func defaultSessionsDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ".pi-go-sessions"
	}
	return filepath.Join(home, ".pi-go", "sessions")
}

func listModels() {
	m := ai.BuiltinModels()
	for _, p := range m.GetProviders() {
		fmt.Printf("%s (%s)\n", p.Name(), p.ID())
		for _, mdl := range p.GetModels() {
			fmt.Printf("  %s\n", mdl.ID)
		}
	}
}

// runtime bundles the shared wiring for both the plain REPL and TUI modes.
type runtime struct {
	models ai.MutableModels
	model  *ai.Model
	env    harness.ExecutionEnv
	sess   *session.Session
	agent  *agent.Agent
	cwd    string
}

func setupRuntime(providerID, modelID, dir string, fresh bool) (*runtime, error) {
	ctx := context.Background()

	models := ai.BuiltinModels()
	model := models.GetModel(providerID, modelID)
	if model == nil {
		return nil, fmt.Errorf("unknown model %s/%s", providerID, modelID)
	}
	if getKey(providerID) == "" {
		fmt.Fprintf(os.Stderr, "warning: no API key found for %q (set %s); request will fail\n", providerID, keyEnvVar(providerID))
	}

	cwd, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	env := harness.NewLocalEnv(cwd)

	sess, err := openSession(ctx, dir, cwd, fresh)
	if err != nil {
		return nil, err
	}

	a := agent.NewAgent(agent.AgentOptions{
		InitialState: &agent.State{
			SystemPrompt: "You are a helpful coding assistant. Use the provided tools to read, edit, and run code as needed.",
			Model:        model,
			Tools:        builtinTools(env),
		},
		StreamFn: models.StreamSimple,
	})

	// Load any prior transcript into the agent state, projecting compaction
	// entries into their summary + retained tail.
	if entries, err := sess.Entries(ctx); err == nil && len(entries) > 0 {
		a.State().Messages = compaction.BuildContext(entries)
	}

	return &runtime{models: models, model: model, env: env, sess: sess, agent: a, cwd: cwd}, nil
}

func run(providerID, modelID, dir string, fresh bool) error {
	ctx := context.Background()
	rt, err := setupRuntime(providerID, modelID, dir, fresh)
	if err != nil {
		return err
	}
	defer fmt.Println()

	// Persist each newly-finalized message and stream assistant text to stdout.
	rt.agent.Subscribe(func(ev agent.Event, cancel context.Context) {
		switch ev.Type {
		case agent.EventMessageEnd:
			if ev.Message != nil {
				_, _ = rt.sess.AppendMessage(ev.Message, ctx)
			}
		case agent.EventMessageUpdate:
			if ev.AssistantMessageEvent != nil && ev.AssistantMessageEvent.Type == ai.EventTextDelta {
				fmt.Print(ev.AssistantMessageEvent.Delta)
			}
		case agent.EventToolExecStart:
			fmt.Fprintf(os.Stderr, "\n[tool] %s ...\n", ev.ToolName)
		case agent.EventToolExecEnd:
			status := "done"
			if ev.IsError {
				status = "error"
			}
			fmt.Fprintf(os.Stderr, "[tool] %s %s\n", ev.ToolName, status)
		}
	})

	fmt.Printf("pi-go agent · %s/%s · session %s\n", rt.model.Provider, rt.model.ID, rt.sess.Metadata().ID)
	fmt.Println("type /help for commands.")

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("you> ")
		line, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println()
			break
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "/") {
			if handleCommand(line, &fresh, &rt.sess, ctx, dir, rt.cwd, rt.agent) {
				continue
			}
			break
		}
		fmt.Print("assistant> ")
		if err := maybeCompact(ctx, rt.agent, rt.sess, rt.models, rt.model); err != nil {
			fmt.Fprintf(os.Stderr, "compaction error: %v\n", err)
		}
		if err := rt.agent.Prompt(line); err != nil {
			fmt.Fprintf(os.Stderr, "prompt error: %v\n", err)
		}
		fmt.Println()
	}
	return nil
}

// runTUI is the terminal-UI mode: a scrolling transcript rendered with
// differential updates and colored styles, using raw key input when available.
func runTUI(providerID, modelID, dir string, fresh bool) error {
	ctx := context.Background()
	rt, err := setupRuntime(providerID, modelID, dir, fresh)
	if err != nil {
		return err
	}

	width, height := tui.TerminalSize()
	if height < 4 {
		height = 8
	}
	if width < 20 {
		width = 80
	}
	screen := tui.NewScreen(os.Stdout)
	defer screen.ShowCursor()
	screen.HideCursor()

	keys, keyErr := tui.NewKeyReader()
	if keyErr == nil {
		defer keys.Close()
	}

	headerStyle := tui.Style{Fg: &tui.Cyan, Bold: true}
	inputStyle := tui.Style{Fg: &tui.Green}

	// Seed the transcript with any prior history.
	lines := []string{}
	if entries, err := rt.sess.Entries(ctx); err == nil && len(entries) > 0 {
		lines = historyLines(compaction.BuildContext(entries))
	}
	var partial strings.Builder
	var input []rune

	render := func() {
		body := make([]string, 0, len(lines)+2)
		body = append(body, lines...)
		if partial.Len() > 0 {
			body = append(body, strings.Split("assistant> "+partial.String(), "\n")...)
		}
		view := make([]string, 0, len(body)+2)
		view = append(view, styleClip(fmt.Sprintf("pi-agent · %s/%s", rt.model.Provider, rt.model.ID), width, headerStyle))
		for _, l := range tui.Bottom(body, height-2) {
			view = append(view, styleClip(l, width, lineStyle(l)))
		}
		if keyErr == nil {
			view = append(view, styleClip("you> "+string(input)+"▏", width, inputStyle))
		}
		screen.Render(view)
	}

	// submit dispatches a completed input line; returns false to exit.
	submit := func(line string) bool {
		input = nil
		screen.Clear()
		line = strings.TrimSpace(line)
		if line == "" {
			render()
			return true
		}
		if strings.HasPrefix(line, "/") {
			switch strings.Fields(line)[0] {
			case "/exit", "/quit":
				return false
			case "/new":
				if s, err := openSession(ctx, dir, rt.cwd, true); err == nil {
					rt.sess = s
					_ = rt.agent.Reset()
					lines = nil
				}
			case "/help":
				fmt.Fprintln(os.Stderr, "/exit  quit · /new  start a new session")
			}
			render()
			return true
		}
		lines = append(lines, "you> "+line)
		partial.Reset()
		render()
		if err := maybeCompact(ctx, rt.agent, rt.sess, rt.models, rt.model); err != nil {
			fmt.Fprintf(os.Stderr, "compaction error: %v\n", err)
		}
		if err := rt.agent.Prompt(line); err != nil {
			lines = append(lines, "error: "+err.Error())
		}
		render()
		return true
	}

	rt.agent.Subscribe(func(ev agent.Event, cancel context.Context) {
		switch ev.Type {
		case agent.EventMessageEnd:
			if ev.Message != nil {
				_, _ = rt.sess.AppendMessage(ev.Message, ctx)
			}
			if partial.Len() > 0 {
				lines = append(lines, "assistant> "+partial.String())
				partial.Reset()
				render()
			}
		case agent.EventMessageUpdate:
			if ev.AssistantMessageEvent != nil && ev.AssistantMessageEvent.Type == ai.EventTextDelta {
				partial.WriteString(ev.AssistantMessageEvent.Delta)
				render()
			}
		case agent.EventToolExecStart:
			lines = append(lines, "[tool] "+ev.ToolName+" ...")
			render()
		case agent.EventToolExecEnd:
			status := "done"
			if ev.IsError {
				status = "error"
			}
			lines = append(lines, "[tool] "+ev.ToolName+" "+status)
			render()
		}
	})

	render()

	// Fallback: line-buffered input when raw mode is unavailable (e.g. piped).
	if keyErr != nil {
		reader := bufio.NewReader(os.Stdin)
		for {
			screen.ShowCursor()
			fmt.Print("\r\nyou> ")
			line, err := reader.ReadString('\n')
			if err != nil {
				fmt.Println()
				return nil
			}
			screen.HideCursor()
			if !submit(line) {
				break
			}
		}
		fmt.Println()
		return nil
	}

	// Raw mode: read keys one at a time with live echo and editing.
	for {
		k, err := keys.ReadKey()
		if err != nil {
			break
		}
		switch {
		case k.Ctrl && k.Rune == 'c':
			return nil
		case k.Code == tui.KeyEnter:
			if !submit(string(input)) {
				fmt.Println()
				return nil
			}
		case k.Code == tui.KeyBackspace:
			if len(input) > 0 {
				input = input[:len(input)-1]
				render()
			}
		case k.Code == tui.KeyRune:
			input = append(input, k.Rune)
			render()
		}
	}
	fmt.Println()
	return nil
}

// handleCommand processes slash commands in plain REPL mode; returns true to
// continue the REPL.
func handleCommand(cmd string, fresh *bool, sess **session.Session, ctx context.Context, dir, cwd string, a *agent.Agent) bool {
	switch strings.Fields(cmd)[0] {
	case "/exit", "/quit":
		return false
	case "/help":
		fmt.Println("/exit  quit\n/new   start a new session\n/models  list models")
	case "/models":
		listModels()
	case "/new":
		s, err := openSession(ctx, dir, cwd, true)
		if err != nil {
			fmt.Fprintln(os.Stderr, "new session error:", err)
			return true
		}
		*sess = s
		_ = a.Reset()
		fmt.Printf("new session %s\n", s.Metadata().ID)
	}
	return true
}

// maybeCompact runs context compaction before a new prompt when the projected
// context exceeds the model window budget.
func maybeCompact(ctx context.Context, a *agent.Agent, sess *session.Session, models ai.MutableModels, model *ai.Model) error {
	entries, err := sess.Entries(ctx)
	if err != nil {
		return err
	}
	settings := compaction.DefaultCompactionSettings
	projected := compaction.BuildContext(entries)
	est := compaction.EstimateContextTokens(projected)
	if !compaction.ShouldCompact(est.Tokens, model.ContextWindow, settings) {
		return nil
	}
	prep := compaction.PrepareCompaction(entries, settings)
	if prep == nil {
		return nil
	}
	res, err := compaction.CompactWithRequest(prep, model, "", "", func(c *ai.Context, opts *ai.SimpleStreamOptions) (*ai.AssistantMessage, error) {
		return models.CompleteSimple(model, c, opts)
	})
	if err != nil {
		return err
	}
	if _, err := sess.AppendCompaction(res.Summary, res.TokensBefore, res.RetainedTail, ctx); err != nil {
		return err
	}
	if entries, err := sess.Entries(ctx); err == nil {
		a.State().Messages = compaction.BuildContext(entries)
	}
	fmt.Fprintf(os.Stderr, "[compacted] %d tokens -> summary (%d retained)\n", res.TokensBefore, len(res.RetainedTail))
	return nil
}

func builtinTools(env harness.ExecutionEnv) []*agent.AgentTool {
	return []*agent.AgentTool{
		harness.CreateReadTool(env, nil),
		harness.CreateEditTool(env),
		harness.CreateWriteTool(env),
		harness.CreateBashTool(env, nil),
	}
}

func openSession(ctx context.Context, dir, cwd string, fresh bool) (*session.Session, error) {
	repo := session.NewJSONLRepo(session.OSFileSystem{}, dir)
	if fresh {
		s, _, err := repo.Create(cwd, ctx)
		return s, err
	}
	metas, err := repo.List(ctx)
	if err != nil {
		return nil, err
	}
	if len(metas) == 0 {
		s, _, err := repo.Create(cwd, ctx)
		return s, err
	}
	return repo.Open(metas[0], ctx)
}

func getKey(providerID string) string {
	for _, name := range keyEnvVars(providerID) {
		if v := os.Getenv(name); v != "" {
			return v
		}
	}
	return ""
}

func keyEnvVars(providerID string) []string {
	switch providerID {
	case "anthropic":
		return []string{"ANTHROPIC_API_KEY", "ANTHROPIC_OAUTH_TOKEN"}
	case "openai":
		return []string{"OPENAI_API_KEY"}
	case "google":
		return []string{"GEMINI_API_KEY"}
	default:
		return []string{strings.ToUpper(providerID) + "_API_KEY"}
	}
}

func keyEnvVar(providerID string) string {
	if v := keyEnvVars(providerID); len(v) > 0 {
		return v[0]
	}
	return ""
}

// messageText extracts the text content from a message.
func messageText(m ai.Message) string {
	var blocks []ai.ContentBlock
	switch v := m.(type) {
	case *ai.UserMessage:
		blocks = v.Content
	case *ai.AssistantMessage:
		blocks = v.Content
	case *ai.ToolResultMessage:
		blocks = v.Content
	}
	var b strings.Builder
	for _, blk := range blocks {
		if blk.Type == ai.ContentTypeText {
			b.WriteString(blk.Text)
		}
	}
	return b.String()
}

// historyLines renders prior messages as display lines for the transcript.
func historyLines(msgs []ai.Message) []string {
	var lines []string
	for _, m := range msgs {
		text := strings.ReplaceAll(messageText(m), "\n", " ")
		if text == "" {
			continue
		}
		switch m.(type) {
		case *ai.UserMessage:
			lines = append(lines, "you> "+text)
		case *ai.AssistantMessage:
			lines = append(lines, "assistant> "+text)
		}
	}
	return lines
}

func clip(s string, width int) string {
	r := []rune(s)
	if len(r) <= width {
		return s
	}
	return string(r[:width])
}

// styleClip truncates plain text to width, then wraps it in the given style.
func styleClip(text string, width int, s tui.Style) string {
	return s.Wrap(clip(text, width))
}

// lineStyle colors a transcript line by its role prefix.
func lineStyle(text string) tui.Style {
	switch {
	case strings.HasPrefix(text, "you> "):
		return tui.Style{Fg: &tui.Green}
	case strings.HasPrefix(text, "[tool] "):
		return tui.Style{Fg: &tui.Yellow}
	case strings.HasPrefix(text, "error"):
		return tui.Style{Fg: &tui.Red}
	default:
		return tui.Style{}
	}
}
