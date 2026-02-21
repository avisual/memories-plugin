# Quick Start

This guide walks through basic usage of avisual memories with Claude Code.

## Prerequisites

- memories installed from source ([Installation Guide](installation.md))
- Ollama running with nomic-embed-text model
- Claude Code (or compatible AI agent)

## Step 1: Verify Setup

```bash
uv run python -m memories diagnose
```

Ensure all checks pass before continuing.

## Step 2: Start Claude Code

Restart Claude Code to pick up the MCP server:

```bash
# If using the CLI
claude-code

# Or restart your IDE/application
```

## Step 3: Store Your First Memory

In Claude Code, ask it to store a memory:

```
Remember this: I prefer pytest over unittest for Python testing.
```

Claude will call:
```python
remember(
    content="User prefers pytest over unittest for Python testing",
    type="preference"
)
```

## Step 4: Store an Antipattern

Tell Claude about a past mistake:

```
Remember as an antipattern: Using rm for file deletion is dangerous. Always use trash() instead so files are recoverable.
```

Claude will call:
```python
remember(
    content="Using rm for file deletion is dangerous. Always use trash() instead so files are recoverable.",
    type="antipattern",
    severity="high",
    instead="Use trash() so files are recoverable",
    importance=0.8
)
```

## Step 5: Recall Memories

Ask Claude to recall context:

```
What testing frameworks do I prefer?
```

Claude will call:
```python
recall(query="testing frameworks")
```

And get back:
```json
{
  "atoms": [{"content": "User prefers pytest over unittest", "score": 0.94, ...}],
  "antipatterns": []
}
```

## Step 6: Test Antipattern Warning

Try an action that matches the antipattern:

```
Delete the file test.txt
```

Claude will:
1. Consider using `rm test.txt`
2. Recall antipatterns about file deletion
3. See the warning: "Using rm is dangerous, use trash()"
4. Choose the safer alternative:

```python
import trash
trash.trash("test.txt")  # File moved to .Trash/, recoverable!
```

## Step 7: Create Connections

Link related memories:

```
Connect my pytest preference to my "use type hints" preference.
```

Claude will call:
```python
connect(
    source_id=1,     # pytest preference
    target_id=5,     # type hints preference
    relationship="elaborates",
    strength=0.7
)
```

Now recalling one will surface the other via spreading activation!

## Step 8: Check Status

```
Show my memory stats.
```

Claude calls:
```python
status()
```

Returns:
```json
{
  "total_atoms": 15,
  "total_synapses": 47,
  "regions": [{"name": "preferences", "count": 8}, ...],
  "db_size_mb": 1.2,
  "ollama_healthy": true
}
```

## Common Workflows

### Research Session

```
# Start research
"I'm researching print-on-demand platforms"

# Claude automatically recalls:
- Past research on POD
- Antipatterns (e.g., "Printful requires $10K minimum")
- Related decisions (e.g., "Chose Printify")

# Store new findings
"Remember: Printify API rate limit is 120 req/min"
```

### Debugging Session

```
# Hit an error
"Remember as antipattern: FastAPI background tasks don't work with SQLite connections - must use connection pool"

# Next time you try background tasks:
# → Antipattern warning appears
# → Claude chooses connection pool approach
# → No repeated mistake!
```

### Project Onboarding

```
# New project
"Recall everything about the e-commerce MVP project"

# Claude gets:
- Architecture decisions
- API keys locations
- Past mistakes (antipatterns)
- Current status
- Related projects
```

## Advanced: Spreading Activation

When you recall one memory, related memories surface automatically:

```
"Recall browser automation best practices"

# Returns:
- "Use 2-4 second pauses between actions" (direct match)
- "Scroll before clicking elements" (connected)
- "Sign in with Apple bypasses bot detection" (related)
- "Peekaboo can capture screenshots for verification" (related tool)
```

No explicit search for each — they flow through the neural graph!

## Memory Types Reference

| Type | Use Case | Example |
|------|----------|---------|
| `fact` | Objective knowledge | "API key stored in Keychain" |
| `experience` | Things that happened | "Launched product Feb 5" |
| `skill` | How to do something | "Use uv for Python package management" |
| `preference` | Your choices | "Prefer pytest over unittest" |
| `insight` | Lessons learned | "Batch API calls to avoid rate limits" |
| `antipattern` | Mistakes to avoid | "Don't use rm, use trash()" |

## Next Steps

- [Configuration](configuration.md) - Customize memory behavior
- [Antipatterns Deep Dive](../concepts/antipatterns.md) - Master mistake prevention
- [Spreading Activation](../concepts/spreading-activation.md) - How retrieval works
- [Best Practices](../guides/best-practices.md) - Tips for effective memory use
- [API Reference](../api/mcp-tools.md) - Full tool documentation
