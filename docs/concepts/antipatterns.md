# Antipatterns

> **The feature that sets avisual memories apart from every other AI memory system.**

## What are Antipatterns?

Antipatterns are memories of **mistakes, failures, and lessons learned** that proactively warn your AI agent _before_ it repeats a past error.

Think of them as your agent's immune system — once burned by a mistake, it develops immunity and warns you before making it again.

## How They Work

### 1. Store the Mistake

When something goes wrong, explicitly store it as an antipattern:

```python
remember(
    content="Use trash instead of rm for file deletion - rm is not recoverable",
    type="antipattern",
    severity="high",
    instead="trash() or mv to .Trash/"
)
```

### 2. Automatic Detection

On every tool use or user prompt, memories:
1. Embeds the current action (e.g., "delete file with os.remove()")
2. Searches antipattern memories for similar past mistakes
3. Surfaces warnings if similarity > 0.75

### 3. Warning Appears

```
⚠️ ANTIPATTERN WARNING

Past mistake: Use trash instead of rm for file deletion
Similarity: 0.92
Context: file deletion, data loss risk

Recommendation: Use trash() or mv to .Trash/ instead
```

Your agent sees this **before executing** and can choose a safer alternative.

## Real-World Examples

### Example 1: Browser Automation

**Mistake:**
```python
# Agent tried to fill web form instantly
browser.fill("#email", "john@example.com")
browser.fill("#password", "secret123")
browser.click("#submit")
# → BLOCKED: Bot detected by website
```

**Stored as antipattern:**
```python
remember(
    content="Browser automation requires human-like behavior: 2-4 second pauses between fields, slow typing, scroll before clicking. Instant form fills get detected as bots.",
    type="antipattern"
)
```

**Next time:**
Agent attempts browser automation → Warning appears → Agent adds delays and human-like behavior → Success!

### Example 2: API Rate Limits

**Mistake:**
```python
# Agent made 100 API calls in 10 seconds
for item in items:
    api.create(item)
# → ERROR: 429 Too Many Requests
```

**Stored as antipattern:**
```python
remember(
    content="Batch API operations hit rate limits. Use batch endpoints or add delays (5 seconds minimum between calls).",
    type="antipattern"
)
```

**Next time:**
Agent sees bulk API task → Warning appears → Agent uses batch endpoint → No rate limit hit!

### Example 3: Data Loss

**Mistake:**
```python
# Agent deleted production database thinking it was test DB
os.system("dropdb production_db")
# → DISASTER: Production data lost
```

**Stored as antipattern:**
```python
remember(
    content="ALWAYS check database name before destructive operations. Use --dry-run flags. Make backups first.",
    type="antipattern",
    importance=1.0  # Maximum importance
)
```

**Next time:**
Agent attempts database deletion → **HIGH-PRIORITY WARNING** → Agent verifies name + creates backup first!

## Why This Matters

### Traditional Systems (No Antipatterns)

```
Agent: "I'll delete this file with rm"
→ Executes rm
→ File gone forever
→ User: "That was important!"
→ Agent: "Sorry, I'll remember for next time"

--- NEXT SESSION ---

Agent: "I'll delete this file with rm"
→ Repeats the same mistake
→ Because memory wasn't structured as preventative warning
```

### With Antipatterns

```
Agent: "I'll delete this file with rm"
→ Antipattern detected (similarity: 0.94)
→ Warning: "Use trash instead - rm not recoverable"
→ Agent: "I'll use trash() instead"
→ File moved to .Trash/ (recoverable)
→ No data loss!
```

## Best Practices

### When to Store Antipatterns

Store as antipattern when:
- ✅ Agent made a mistake that caused problems
- ✅ Action was blocked by external system (bot detection, rate limit)
- ✅ You want to prevent specific behavior in future
- ✅ Lesson learned from debugging session

Don't store as antipattern:
- ❌ Expected errors (404s, network timeouts)
- ❌ User errors (typos, wrong input)
- ❌ Preference changes (use `type="preference"`)

### Writing Effective Antipatterns

**Bad antipattern:**
```python
remember(
    content="Error happened",  # Too vague
    type="antipattern"
)
```

**Good antipattern:**
```python
remember(
    content="Gumroad browser automation causes text corruption. Use manual upload instead or wait for API access. Robot detection is aggressive.",
    type="antipattern",
    importance=0.8
)
```

**Components of a good antipattern:**
1. **What went wrong** - Specific action/context
2. **Why it failed** - Root cause
3. **How to avoid** - Concrete alternative
4. **When it applies** - Triggering conditions

### Importance Levels

Use `importance` to prioritize warnings:

```python
# Critical (1.0) - Data loss, security issues
remember(
    content="Never commit API keys to git",
    type="antipattern",
    importance=1.0
)

# High (0.8) - Blocked actions, failed automations
remember(
    content="Etsy blocks automation without 3-second pauses",
    type="antipattern",
    importance=0.8
)

# Medium (0.5) - Inefficient approaches
remember(
    content="Prefer API over browser automation when available",
    type="antipattern",
    importance=0.5
)

# Low (0.3) - Style preferences
remember(
    content="User prefers pytest over unittest",
    type="antipattern",  # Better as "preference" type
    importance=0.3
)
```

## How Similarity Matching Works

Antipatterns use **semantic similarity** (embeddings) to detect related actions:

```python
# Stored antipattern:
"Use trash instead of rm for file deletion"

# Triggers on:
"os.remove('file.txt')"           # similarity: 0.89
"subprocess.run(['rm', 'file'])"  # similarity: 0.91  
"shutil.rmtree('/data')"          # similarity: 0.82

# Does NOT trigger on:
"mkdir new_folder"                # similarity: 0.12
"cat file.txt"                    # similarity: 0.31
```

Threshold: **0.75 similarity** (configurable in settings)

## Integration with Other Memory Types

Antipatterns work alongside other memory types:

```python
# Decision (what you chose)
remember(
    content="Chose Printify over Printful for sticker products",
    type="decision"
)

# Antipattern (what NOT to do)
remember(
    content="Don't use Printful API - requires $10K minimum volume",
    type="antipattern"
)

# Fact (neutral knowledge)
remember(
    content="Printify API key stored in Keychain as printify-api-token",
    type="fact"
)
```

When agent considers print-on-demand task:
- Recalls **decision** → "Use Printify"
- Recalls **antipattern** → "Avoid Printful"
- Recalls **fact** → "API key location"

## Viewing Antipatterns

Search antipatterns via the MCP tool:
```python
recall(query="browser automation", types=["antipattern"])
```

Or use the eval CLI to see what would surface for a given prompt:
```bash
uv run python -m memories eval "delete a file" --verbose
```

## Performance Impact

Antipattern checking adds minimal overhead:
- **< 10ms** per tool use (embedding + similarity search)
- Runs asynchronously (doesn't block agent)
- Uses local Ollama (no API costs)

## Future Enhancements

Roadmap for antipatterns:
- [ ] Severity levels (warning vs blocking)
- [ ] Expiration (auto-remove after N months)
- [ ] Pattern detection (auto-create from repeated errors)
- [ ] Confidence scores (learn which warnings are heeded)
- [ ] Categories (security, data-loss, rate-limits, etc.)

## Summary

Antipatterns are:
- ✅ **Proactive** - warn before mistakes
- ✅ **Contextual** - semantic similarity matching
- ✅ **Explicit** - you control what gets stored
- ✅ **Fast** - < 10ms overhead
- ✅ **Unique** - no other memory system has this

**Bottom line:** They save hours of debugging and prevent data loss. Use them liberally!

## Next Steps

- [Spreading Activation](spreading-activation.md) - How memories are retrieved
- [Hebbian Learning](hebbian-learning.md) - How connections strengthen
- [Best Practices](../guides/best-practices.md) - Tips for effective memory use
