package ai

import "context"

// CompletionOptions configures an AI completion request
type CompletionOptions struct {
	// Model specifies which model to use (e.g., "sonnet", "haiku", "opus", or full model ID)
	// Empty string uses the backend's default model
	Model string

	// MaxTokens limits the response length
	// Zero uses a sensible default (4096)
	MaxTokens int

	// JSONSchema enables structured output validation (Claude Code CLI only)
	// When set, the backend will use --json-schema flag for structured output
	JSONSchema string

	// Operation names this request for logging and metrics
	Operation string
}

// AIBackend abstracts AI completion calls, enabling pluggable backends
// (Anthropic SDK vs Claude Code CLI)
type AIBackend interface {
	// Complete sends a prompt and returns the response text
	Complete(ctx context.Context, prompt string, opts CompletionOptions) (string, error)

	// HealthCheck verifies the backend is operational
	HealthCheck(ctx context.Context) error

	// Model returns the default model name for this backend
	Model() string

	// Name returns the backend identifier (e.g., "anthropic-sdk", "claude-code")
	Name() string

	// SupportsJSONSchema returns true if this backend supports structured output
	// via JSON schema validation
	SupportsJSONSchema() bool
}

// DefaultCompletionOptions returns sensible defaults for a completion request
func DefaultCompletionOptions(operation string) CompletionOptions {
	return CompletionOptions{
		MaxTokens: 4096,
		Operation: operation,
	}
}

// WithModel returns a copy of the options with the model set
func (o CompletionOptions) WithModel(model string) CompletionOptions {
	o.Model = model
	return o
}

// WithMaxTokens returns a copy of the options with max tokens set
func (o CompletionOptions) WithMaxTokens(maxTokens int) CompletionOptions {
	o.MaxTokens = maxTokens
	return o
}

// WithJSONSchema returns a copy of the options with JSON schema set
func (o CompletionOptions) WithJSONSchema(schema string) CompletionOptions {
	o.JSONSchema = schema
	return o
}
