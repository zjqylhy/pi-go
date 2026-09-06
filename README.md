# pi-go

用 Go 从零实现的 coding agent 运行时，移植自 TypeScript 的 `pi` 项目的核心模块。

## 模块

| 包 | 职责 |
|---|---|
| `ai` | LLM provider 抽象 + SSE 流式（Anthropic / OpenAI / Google / DeepSeek / Groq / xAI / OpenRouter / Mistral / Moonshot / Zhipu / Kimi / Together） |
| `agent` | agent 运行循环：流式回复、工具调用（顺序/并行）、steering / follow-up 队列 |
| `harness` | 内置工具：`read` / `write` / `edit` / `bash`（文件系统 + shell 抽象） |
| `session` | 会话存储：内存 + JSONL 持久化、父链构建、usage 聚合 |
| `compaction` | 上下文压缩：token 估算、切点选择、摘要生成 |
| `tui` | 终端 UI 库：差分渲染、彩色 Style、按键输入（Windows raw mode） |

## 命令

```bash
# 列出 provider / model
go run ./cmd/pi-ai -models

# 一次性补全（需对应 provider 的 API Key）
go run ./cmd/pi-ai -provider anthropic -model claude-sonnet-4-5 "你好"

# 交互式 coding agent（恢复最近会话）
go run ./cmd/pi-agent

# 终端 UI 模式（差分渲染 + 彩色 + raw 按键输入）
go run ./cmd/pi-agent -tui

# TUI 库 demo
go run ./cmd/pi-tui
```

## 环境变量

按 provider 设置对应 API Key，例如 `ANTHROPIC_API_KEY`、`OPENAI_API_KEY`、`GEMINI_API_KEY`。

## 构建与测试

```bash
go build ./...
go test ./...
```

模块路径：`github.com/zjqylhy/pi-go`。