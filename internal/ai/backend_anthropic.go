package ai

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// AnthropicSDKBackend implements AIBackend using the Anthropic SDK
type AnthropicSDKBackend struct {
	client *anthropic.Client
	model  string
}

// Compile-time check that AnthropicSDKBackend implements AIBackend
var _ AIBackend = (*AnthropicSDKBackend)(nil)

// AnthropicBackendConfig configures the Anthropic SDK backend
type AnthropicBackendConfig struct {
	// APIKey is the Anthropic API key (if empty, reads from ANTHROPIC_API_KEY env var)
	APIKey string
	// Model is the default model to use (default: claude-sonnet-4-5-20250929)
	Model string
}

// NewAnthropicSDKBackend creates a new Anthropic SDK backend
func NewAnthropicSDKBackend(cfg *AnthropicBackendConfig) (*AnthropicSDKBackend, error) {
	apiKey := cfg.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("ANTHROPIC_API_KEY not set")
		}
	}

	model := cfg.Model
	if model == "" {
		model = GetDefaultModel()
	}

	client := anthropic.NewClient(option.WithAPIKey(apiKey))

	return &AnthropicSDKBackend{
		client: &client,
		model:  model,
	}, nil
}

// Complete sends a prompt and returns the response text
func (b *AnthropicSDKBackend) Complete(ctx context.Context, prompt string, opts CompletionOptions) (string, error) {
	model := opts.Model
	if model == "" {
		model = b.model
	}

	maxTokens := opts.MaxTokens
	if maxTokens == 0 {
		maxTokens = 4096
	}

	resp, err := b.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: int64(maxTokens),
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(prompt)),
		},
	})
	if err != nil {
		return "", fmt.Errorf("anthropic API call failed: %w", err)
	}

	// Extract text content from the response
	var result strings.Builder
	for _, block := range resp.Content {
		if block.Type == "text" {
			result.WriteString(block.Text)
		}
	}

	return result.String(), nil
}

// CompleteWithUsage sends a prompt and returns both the response text and usage stats
// This is used by the Supervisor to track token usage for cost attribution
func (b *AnthropicSDKBackend) CompleteWithUsage(ctx context.Context, prompt string, opts CompletionOptions) (string, int64, int64, error) {
	model := opts.Model
	if model == "" {
		model = b.model
	}

	maxTokens := opts.MaxTokens
	if maxTokens == 0 {
		maxTokens = 4096
	}

	resp, err := b.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: int64(maxTokens),
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(prompt)),
		},
	})
	if err != nil {
		return "", 0, 0, fmt.Errorf("anthropic API call failed: %w", err)
	}

	// Extract text content from the response
	var result strings.Builder
	for _, block := range resp.Content {
		if block.Type == "text" {
			result.WriteString(block.Text)
		}
	}

	return result.String(), resp.Usage.InputTokens, resp.Usage.OutputTokens, nil
}

// HealthCheck verifies the backend is operational
func (b *AnthropicSDKBackend) HealthCheck(ctx context.Context) error {
	// Make a minimal API call to verify connectivity
	// Use a very short prompt to minimize cost
	_, err := b.Complete(ctx, "Hi", CompletionOptions{
		MaxTokens: 10,
		Operation: "health_check",
	})
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	return nil
}

// Model returns the default model name
func (b *AnthropicSDKBackend) Model() string {
	return b.model
}

// Name returns the backend identifier
func (b *AnthropicSDKBackend) Name() string {
	return "anthropic-sdk"
}

// SupportsJSONSchema returns false as the SDK doesn't support JSON schema validation
// (would need to use Claude's native structured output, not implemented here)
func (b *AnthropicSDKBackend) SupportsJSONSchema() bool {
	return false
}
