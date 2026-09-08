package llm

import (
	"context"
	"fmt"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// Config holds configuration for the Claude client.
type Config struct {
	APIKey      string
	Model       string
	MaxTokens   int
	Temperature float64
}

// DefaultConfig returns a default configuration.
func DefaultConfig() Config {
	return Config{
		APIKey:      os.Getenv("ANTHROPIC_API_KEY"),
		Model:       "claude-sonnet-4-20250514",
		MaxTokens:   4096,
		Temperature: 0.7,
	}
}

// Client wraps the Anthropic SDK for content generation.
type Client struct {
	client *anthropic.Client
	config Config
}

// NewClient creates a new Claude client with the given configuration.
func NewClient(cfg Config) (*Client, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("ANTHROPIC_API_KEY is required")
	}

	client := anthropic.NewClient(option.WithAPIKey(cfg.APIKey))

	return &Client{
		client: &client,
		config: cfg,
	}, nil
}

// Generate sends a prompt to Claude and returns the response.
func (c *Client) Generate(ctx context.Context, systemPrompt, userPrompt string) (string, error) {
	params := anthropic.MessageNewParams{
		Model:     c.config.Model,
		MaxTokens: int64(c.config.MaxTokens),
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(userPrompt)),
		},
	}

	if systemPrompt != "" {
		params.System = []anthropic.TextBlockParam{
			{
				Type: "text",
				Text: systemPrompt,
			},
		}
	}

	message, err := c.client.Messages.New(ctx, params)
	if err != nil {
		return "", fmt.Errorf("claude API error: %w", err)
	}

	// Extract text from response
	var result string
	for _, block := range message.Content {
		if block.Type == "text" {
			result += block.Text
		}
	}

	return result, nil
}

// GenerateWithRetry attempts generation with retries on failure.
func (c *Client) GenerateWithRetry(ctx context.Context, systemPrompt, userPrompt string, maxRetries int) (string, error) {
	var lastErr error
	for i := 0; i < maxRetries; i++ {
		result, err := c.Generate(ctx, systemPrompt, userPrompt)
		if err == nil {
			return result, nil
		}
		lastErr = err
	}
	return "", fmt.Errorf("failed after %d retries: %w", maxRetries, lastErr)
}
