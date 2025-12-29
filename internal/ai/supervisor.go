package ai

import (
	"context"
	"fmt"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/steveyegge/vc/internal/storage"
	"github.com/steveyegge/vc/internal/types"
	"golang.org/x/sync/semaphore"
)

// AI Model Constants (vc-35: Tiered AI Model Strategy)
//
// VC uses a tiered approach to AI model selection based on task complexity:
// - Sonnet 4.5: Complex tasks requiring deep reasoning (assessment, analysis, planning)
// - Haiku: Simple tasks like file size checks, commit messages, cruft detection
//
// Cost savings: Haiku is ~80% cheaper than Sonnet for simple operations.
//
// Environment variable overrides (vc-lf8j Phase 2):
// - VC_MODEL_DEFAULT: Override default model (default: Sonnet)
// - VC_MODEL_SIMPLE: Override model for simple tasks (default: Haiku)
const (
	// ModelSonnet is the high-end model for complex reasoning tasks
	ModelSonnet = "claude-sonnet-4-5-20250929"

	// ModelHaiku is the cost-efficient model for simple tasks
	ModelHaiku = "claude-3-5-haiku-20241022"
)

// GetDefaultModel returns the default model, checking VC_MODEL_DEFAULT env var first
func GetDefaultModel() string {
	if model := os.Getenv("VC_MODEL_DEFAULT"); model != "" {
		return model
	}
	return ModelSonnet
}

// GetSimpleTaskModel returns the model for simple tasks, checking VC_MODEL_SIMPLE env var first
func GetSimpleTaskModel() string {
	if model := os.Getenv("VC_MODEL_SIMPLE"); model != "" {
		return model
	}
	return ModelHaiku
}

// BackendType represents the type of AI backend to use
type BackendType string

const (
	// BackendAnthropicSDK uses the Anthropic SDK directly (requires API key)
	BackendAnthropicSDK BackendType = "anthropic-sdk"
	// BackendClaudeCode uses the Claude Code CLI (uses subscription)
	BackendClaudeCode BackendType = "claude-code"
)

// Supervisor handles AI-powered assessment and analysis of issues
// It also implements the MissionPlanner interface for mission orchestration
//
// The Supervisor's responsibilities are distributed across multiple files:
// - supervisor.go: Core struct and constructor (this file)
// - retry.go: Circuit breaker and retry logic
// - assessment.go: Pre-execution assessment and completion assessment
// - analysis.go: Post-execution analysis
// - recovery.go: Quality gate failure recovery strategies
// - code_review.go: Code quality and test coverage analysis
// - deduplication.go: Duplicate issue detection
// - translation.go: Discovered issue creation
// - planning.go: Mission planning and phase refinement
// - utils.go: Shared utilities (logging, summarization, truncation)
type Supervisor struct {
	client         *anthropic.Client
	backend        AIBackend           // Pluggable AI backend (vc-dtqf)
	store          storage.Storage
	model          string
	retry          RetryConfig
	circuitBreaker *CircuitBreaker
	concurrencySem *semaphore.Weighted // Limits concurrent AI API calls (vc-220)
	costTracker    CostTracker         // Tracks AI costs and enforces budgets (vc-e3s7)
}

// Compile-time check that Supervisor implements MissionPlanner
var _ types.MissionPlanner = (*Supervisor)(nil)

// CostTracker defines the interface for cost budgeting (vc-e3s7, vc-7e21)
// This allows dependency injection and testing without circular imports
type CostTracker interface {
	// RecordUsage records token usage for an issue
	// Returns budget status (as interface{} to avoid circular dependencies) and error
	RecordUsage(ctx context.Context, issueID string, inputTokens, outputTokens int64) (interface{}, error)
	// CanProceed checks if we can make another AI call within budget
	CanProceed(issueID string) (bool, string)
	// RecordOperation records a detailed quota operation for cost attribution (vc-7e21)
	// This is optional - if quota monitoring is disabled, this is a no-op
	// Takes a generic interface{} to avoid importing the cost package (circular dependency)
	RecordOperation(ctx context.Context, op interface{}) error
}

// Config holds supervisor configuration
type Config struct {
	APIKey      string          // Anthropic API key (if empty, reads from ANTHROPIC_API_KEY env var)
	Model       string          // Model to use (default: claude-sonnet-4-5-20250929)
	Store       storage.Storage
	Retry       RetryConfig     // Retry configuration (uses defaults if not specified)
	CostTracker CostTracker     // Optional cost tracker for budget enforcement (vc-e3s7)
	Backend     BackendType     // AI backend type (default: anthropic-sdk, or from VC_AI_BACKEND env var)
	CLIPath     string          // Path to claude CLI (only used when Backend is claude-code)
}

// NewSupervisor creates a new AI supervisor
func NewSupervisor(cfg *Config) (*Supervisor, error) {
	if cfg.Store == nil {
		return nil, fmt.Errorf("storage is required")
	}

	model := cfg.Model
	if model == "" {
		model = GetDefaultModel()
	}

	// Use default retry config if not specified
	retry := cfg.Retry
	if retry.MaxRetries == 0 {
		retry = DefaultRetryConfig()
	}

	// Determine backend type from config or environment
	backendType := cfg.Backend
	if backendType == "" {
		if envBackend := os.Getenv("VC_AI_BACKEND"); envBackend != "" {
			backendType = BackendType(envBackend)
		} else {
			backendType = BackendAnthropicSDK // Default to SDK
		}
	}

	// Create the appropriate backend
	var backend AIBackend
	var client *anthropic.Client

	switch backendType {
	case BackendClaudeCode:
		// Create Claude Code CLI backend
		cliBackend, err := NewClaudeCodeCLIBackend(&ClaudeCodeBackendConfig{
			Model:   model,
			CLIPath: cfg.CLIPath,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create Claude Code backend: %w", err)
		}
		backend = cliBackend
		fmt.Printf("AI backend initialized: claude-code (model=%s)\n", model)

	case BackendAnthropicSDK, "": // Default to SDK
		// Get API key for SDK backend
		apiKey := cfg.APIKey
		if apiKey == "" {
			apiKey = os.Getenv("ANTHROPIC_API_KEY")
			if apiKey == "" {
				return nil, fmt.Errorf("ANTHROPIC_API_KEY not set (required for anthropic-sdk backend)")
			}
		}

		// Create SDK backend
		sdkBackend, err := NewAnthropicSDKBackend(&AnthropicBackendConfig{
			APIKey: apiKey,
			Model:  model,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create Anthropic SDK backend: %w", err)
		}
		backend = sdkBackend

		// Also create the raw client for backward compatibility with CallAPI
		rawClient := anthropic.NewClient(option.WithAPIKey(apiKey))
		client = &rawClient
		fmt.Printf("AI backend initialized: anthropic-sdk (model=%s)\n", model)

	default:
		return nil, fmt.Errorf("unknown backend type: %s (valid options: anthropic-sdk, claude-code)", backendType)
	}

	// Initialize circuit breaker if enabled
	var circuitBreaker *CircuitBreaker
	if retry.CircuitBreakerEnabled {
		circuitBreaker = NewCircuitBreaker(
			retry.FailureThreshold,
			retry.SuccessThreshold,
			retry.OpenTimeout,
		)
		fmt.Printf("Circuit breaker initialized: threshold=%d failures, recovery=%d successes, timeout=%v\n",
			retry.FailureThreshold, retry.SuccessThreshold, retry.OpenTimeout)
	}

	// Initialize concurrency limiter (vc-220)
	var concurrencySem *semaphore.Weighted
	if retry.MaxConcurrentCalls > 0 {
		concurrencySem = semaphore.NewWeighted(int64(retry.MaxConcurrentCalls))
		fmt.Printf("AI concurrency limiter initialized: max_concurrent=%d calls\n", retry.MaxConcurrentCalls)
	}

	return &Supervisor{
		client:         client, // May be nil if using Claude Code backend
		backend:        backend,
		store:          cfg.Store,
		model:          model,
		retry:          retry,
		circuitBreaker: circuitBreaker,
		concurrencySem: concurrencySem,
		costTracker:    cfg.CostTracker, // Optional cost tracker (vc-e3s7)
	}, nil
}

// HealthCheck performs a pre-flight check of the supervisor's health
// Returns an error if the circuit breaker is open or if there are API connectivity issues
func (s *Supervisor) HealthCheck(ctx context.Context) error {
	// Check circuit breaker state
	if s.circuitBreaker != nil {
		state, failures, _ := s.circuitBreaker.GetMetrics()
		switch state {
		case CircuitOpen:
			return fmt.Errorf("AI supervisor unavailable: %w (failures=%d, retry in %v)",
				ErrCircuitOpen, failures, s.retry.OpenTimeout)
		case CircuitHalfOpen:
			// Allow execution in half-open state (probing for recovery)
			fmt.Printf("AI supervisor in half-open state (probing for recovery)\n")
		case CircuitClosed:
			// Normal operation
		}
	}
	return nil
}

// CompleteWithBackend executes a completion using the configured AI backend.
// This is the preferred method for new code - it uses the pluggable backend
// and handles retry, circuit breaker, and concurrency limiting.
//
// Returns the response text.
func (s *Supervisor) CompleteWithBackend(ctx context.Context, prompt string, opts CompletionOptions) (string, error) {
	if opts.Model == "" {
		opts.Model = s.model
	}
	if opts.MaxTokens == 0 {
		opts.MaxTokens = 4096
	}

	var responseText string
	err := s.retryWithBackoff(ctx, opts.Operation, func(attemptCtx context.Context) error {
		text, backendErr := s.backend.Complete(attemptCtx, prompt, opts)
		if backendErr != nil {
			return backendErr
		}
		responseText = text
		return nil
	})

	if err != nil {
		return "", fmt.Errorf("%s failed: %w", opts.Operation, err)
	}

	return responseText, nil
}

// GetBackend returns the configured AI backend
func (s *Supervisor) GetBackend() AIBackend {
	return s.backend
}

// CallAPI makes a raw API call to the Anthropic API with the given prompt.
// This is a low-level method for use by specialized components that need
// direct access to the API (e.g., convergence detection).
//
// DEPRECATED: Use CompleteWithBackend for new code. This method requires the
// Anthropic SDK backend and will not work with Claude Code CLI backend.
//
// Use the higher-level methods (Assess, Analyze, etc.) for standard workflows.
func (s *Supervisor) CallAPI(ctx context.Context, prompt string, model string, maxTokens int) (*anthropic.Message, error) {
	// Check if we have a raw client (SDK backend)
	if s.client == nil {
		return nil, fmt.Errorf("CallAPI requires anthropic-sdk backend (client is nil)")
	}

	if model == "" {
		model = s.model
	}
	if maxTokens == 0 {
		maxTokens = 4096
	}

	// Respect concurrency limits (vc-220)
	if s.concurrencySem != nil {
		if err := s.concurrencySem.Acquire(ctx, 1); err != nil {
			return nil, fmt.Errorf("failed to acquire AI concurrency slot: %w", err)
		}
		defer s.concurrencySem.Release(1)
	}

	var response *anthropic.Message
	err := s.retryWithBackoff(ctx, "api_call", func(attemptCtx context.Context) error {
		resp, apiErr := s.client.Messages.New(attemptCtx, anthropic.MessageNewParams{
			Model:     anthropic.Model(model),
			MaxTokens: int64(maxTokens),
			Messages: []anthropic.MessageParam{
				anthropic.NewUserMessage(anthropic.NewTextBlock(prompt)),
			},
		})
		if apiErr != nil {
			return apiErr
		}
		response = resp
		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("anthropic API call failed: %w", err)
	}

	return response, nil
}
