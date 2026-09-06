// Command pi-tui demonstrates the differential-rendering TUI: a scrolling
// transcript viewport that only redraws changed rows, plus colored styles and
// (when available) raw keyboard input.
package main

import (
	"fmt"
	"os"
	"time"

	"github.com/zjqylhy/pi-go/tui"
)

func main() {
	runScrollDemo()
	runKeyDemo()
}

func runScrollDemo() {
	_, h := tui.TerminalSize()
	if h < 4 {
		h = 8
	}

	screen := tui.NewScreen(os.Stdout)
	screen.HideCursor()
	defer screen.ShowCursor()

	header := tui.Style{Fg: &tui.Cyan, Bold: true}.Wrap("pi-tui demo · scrolling transcript")
	msgStyle := tui.Style{Fg: &tui.Green}
	transcript := []string{}
	for i := 1; i <= 20; i++ {
		transcript = append(transcript, msgStyle.Wrap(fmt.Sprintf("  message %02d: the agent responded with some text", i)))
		frame := make([]string, 0, len(transcript)+2)
		frame = append(frame, header)
		frame = append(frame, tui.Bottom(transcript, h-2)...)
		frame = append(frame, tui.Style{Fg: &tui.Yellow}.Wrap("> "))
		screen.Render(frame)
		time.Sleep(120 * time.Millisecond)
	}
}

func runKeyDemo() {
	r, err := tui.NewKeyReader()
	if err != nil {
		fmt.Fprintln(os.Stderr, "key demo skipped:", err)
		return
	}
	defer r.Close()

	fmt.Println("\npress keys (q to quit):")
	keyStyle := tui.Style{Fg: &tui.Yellow, Bold: true}
	for i := 0; i < 30; i++ {
		k, err := r.ReadKey()
		if err != nil {
			fmt.Fprintln(os.Stderr, "read error:", err)
			return
		}
		fmt.Printf("  %s\n", keyStyle.Wrap(fmt.Sprintf("code=%d rune=%q ctrl=%v alt=%v shift=%v", k.Code, k.Rune, k.Ctrl, k.Alt, k.Shift)))
		if k.Code == tui.KeyRune && k.Rune == 'q' {
			return
		}
	}
}
