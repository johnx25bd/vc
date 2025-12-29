package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
)

// ClaudeCodeCLIBackend implements AIBackend using the Claude Code CLI
type ClaudeCodeCLIBackend struct {
	model      string
	cliPath    string // Path to claude CLI (default: "claude")
	workingDir string // Working directory for CLI (optional)
}

// Compile-time check that ClaudeCodeCLIBackend implements AIBackend
var _ AIBackend = (*ClaudeCodeCLIBackend)(nil)

// ClaudeCodeBackendConfig configures the Claude Code CLI backend
type ClaudeCodeBackendConfig struct {
	// Model is the default model to use (e.g., "sonnet", "haiku", "opus")
	// Default: "sonnet"
	Model string
	// CLIPath is the path to the claude CLI binary
	// Default: "claude" (found via PATH)
	CLIPath string
	// WorkingDir is the working directory for CLI execution
	// Default: current directory
	WorkingDir string
}

// claudeResponse represents the JSON response from claude -p --output-format json
type claudeResponse struct {
	Type          string  `json:"type"`
	Subtype       string  `json:"subtype"`
	IsError       bool    `json:"is_error"`
	Result        string  `json:"result"`
	DurationMS    int64   `json:"duration_ms"`
	DurationAPIMS int64   `json:"duration_api_ms"`
	NumTurns      int     `json:"num_turns"`
	TotalCostUSD  float64 `json:"total_cost_usd"`
	SessionID     string  `json:"session_id"`
	Usage         *struct {
		InputTokens              int64 `json:"input_tokens"`
		OutputTokens             int64 `json:"output_tokens"`
		CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
		CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
	} `json:"usage,omitempty"`
}

// NewClaudeCodeCLIBackend creates a new Claude Code CLI backend
func NewClaudeCodeCLIBackend(cfg *ClaudeCodeBackendConfig) (*ClaudeCodeCLIBackend, error) {
	model := cfg.Model
	if model == "" {
		model = "sonnet" // Default to sonnet for supervision tasks
	}

	cliPath := cfg.CLIPath
	if cliPath == "" {
		cliPath = "claude"
	}

	// Verify the CLI is available
	_, err := exec.LookPath(cliPath)
	if err != nil {
		return nil, fmt.Errorf("claude CLI not found at %q: %w (try running 'npm install -g @anthropic-ai/claude-code')", cliPath, err)
	}

	return &ClaudeCodeCLIBackend{
		model:      model,
		cliPath:    cliPath,
		workingDir: cfg.WorkingDir,
	}, nil
}

// Complete sends a prompt and returns the response text
func (b *ClaudeCodeCLIBackend) Complete(ctx context.Context, prompt string, opts CompletionOptions) (string, error) {
	// Build command arguments
	args := []string{
		"-p",                  // Print mode (non-interactive)
		"--output-format", "json", // JSON output for parsing
		"--tools", "",         // Disable tools for pure completion
	}

	// Add model if specified
	model := opts.Model
	if model == "" {
		model = b.model
	}
	args = append(args, "--model", model)

	// Add JSON schema if specified (for structured output)
	if opts.JSONSchema != "" {
		args = append(args, "--json-schema", opts.JSONSchema)
	}

	// Add the prompt
	args = append(args, prompt)

	// Create the command
	cmd := exec.CommandContext(ctx, b.cliPath, args...)
	if b.workingDir != "" {
		cmd.Dir = b.workingDir
	}

	// Capture stderr for error messages
	var stderr strings.Builder
	cmd.Stderr = &stderr

	// Execute and capture output
	output, err := cmd.Output()
	if err != nil {
		// Include stderr in error message for debugging
		stderrStr := stderr.String()
		if stderrStr != "" {
			return "", fmt.Errorf("claude CLI failed: %w\nstderr: %s", err, stderrStr)
		}
		return "", fmt.Errorf("claude CLI failed: %w", err)
	}

	// Parse the JSON response
	var resp claudeResponse
	if err := json.Unmarshal(output, &resp); err != nil {
		// Show truncated output for debugging
		outputStr := string(output)
		if len(outputStr) > 500 {
			outputStr = outputStr[:500] + "... (truncated)"
		}
		return "", fmt.Errorf("failed to parse claude CLI response: %w\noutput: %s", err, outputStr)
	}

	// Check for error response
	if resp.IsError {
		return "", fmt.Errorf("claude CLI returned error: %s (subtype: %s)", resp.Result, resp.Subtype)
	}

	// Validate response type
	if resp.Type != "result" {
		return "", fmt.Errorf("unexpected response type: %s (expected 'result')", resp.Type)
	}

	return resp.Result, nil
}

// CompleteWithUsage sends a prompt and returns both the response text and usage stats
// Note: Claude Code CLI aggregates usage across internal calls, so this may not be
// as accurate as direct API usage tracking
func (b *ClaudeCodeCLIBackend) CompleteWithUsage(ctx context.Context, prompt string, opts CompletionOptions) (string, int64, int64, error) {
	// Build command arguments
	args := []string{
		"-p",
		"--output-format", "json",
		"--tools", "",
	}

	model := opts.Model
	if model == "" {
		model = b.model
	}
	args = append(args, "--model", model)

	if opts.JSONSchema != "" {
		args = append(args, "--json-schema", opts.JSONSchema)
	}

	args = append(args, prompt)

	cmd := exec.CommandContext(ctx, b.cliPath, args...)
	if b.workingDir != "" {
		cmd.Dir = b.workingDir
	}

	var stderr strings.Builder
	cmd.Stderr = &stderr

	output, err := cmd.Output()
	if err != nil {
		stderrStr := stderr.String()
		if stderrStr != "" {
			return "", 0, 0, fmt.Errorf("claude CLI failed: %w\nstderr: %s", err, stderrStr)
		}
		return "", 0, 0, fmt.Errorf("claude CLI failed: %w", err)
	}

	var resp claudeResponse
	if err := json.Unmarshal(output, &resp); err != nil {
		return "", 0, 0, fmt.Errorf("failed to parse claude CLI response: %w", err)
	}

	if resp.IsError {
		return "", 0, 0, fmt.Errorf("claude CLI returned error: %s", resp.Result)
	}

	// Extract usage if available
	var inputTokens, outputTokens int64
	if resp.Usage != nil {
		inputTokens = resp.Usage.InputTokens + resp.Usage.CacheCreationInputTokens + resp.Usage.CacheReadInputTokens
		outputTokens = resp.Usage.OutputTokens
	}

	return resp.Result, inputTokens, outputTokens, nil
}

// HealthCheck verifies the backend is operational
func (b *ClaudeCodeCLIBackend) HealthCheck(ctx context.Context) error {
	// Check if CLI is available
	cmd := exec.CommandContext(ctx, b.cliPath, "--version")
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("claude CLI health check failed: %w", err)
	}

	// Verify we got a version string
	version := strings.TrimSpace(string(output))
	if version == "" {
		return fmt.Errorf("claude CLI returned empty version")
	}

	// Log for debugging
	if os.Getenv("VC_DEBUG_BACKENDS") != "" {
		fmt.Printf("Claude Code CLI backend healthy: version %s\n", version)
	}

	return nil
}

// Model returns the default model name
func (b *ClaudeCodeCLIBackend) Model() string {
	return b.model
}

// Name returns the backend identifier
func (b *ClaudeCodeCLIBackend) Name() string {
	return "claude-code"
}

// SupportsJSONSchema returns true as the CLI supports --json-schema flag
func (b *ClaudeCodeCLIBackend) SupportsJSONSchema() bool {
	return true
}
