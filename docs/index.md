# avisual memories

> **The only AI memory system that prevents mistakes _before_ they happen.**

## What is avisual memories?

avisual memories is a brain-like memory system for AI agents that provides:

- **Persistent long-term memory** across sessions
- **Neural-inspired retrieval** via spreading activation
- **Antipatterns** that warn agents before repeating mistakes
- **Hebbian learning** that automatically strengthens connections
- **Memory consolidation** that decays, merges, and prunes old knowledge
- **Local-first design** with no API costs (uses Ollama)

## Why avisual memories?

Unlike basic RAG systems or commercial memory APIs, avisual memories is built like an actual brain:

| Feature | avisual memories | Mem0 | LangChain | ChromaDB |
|---------|------------------|------|-----------|----------|
| **Antipatterns** | ✅ | ❌ | ❌ | ❌ |
| Spreading activation | ✅ | ❌ | ❌ | ❌ |
| Hebbian learning | ✅ | ❌ | ❌ | ❌ |
| Memory consolidation | ✅ | ❌ | ❌ | ❌ |
| Local-first | ✅ | ❌ | ✅ | ✅ |
| Open source (MIT) | ✅ | ❌ | ✅ | ✅ |
| Cost | **Free** | $249/mo | Free | Free |

## Quick Start

```bash
# Install
pip install avisual-memories

# Setup (interactive)
avisual-memories setup --interactive

# Verify
avisual-memories diagnose
```

Then restart your AI agent (Claude Code, Aider, etc.) and it will have persistent memory!

## Key Features

### 🚨 Antipatterns

The killer feature. When your agent tries to repeat a past mistake, memories surfaces a warning _before_ the command runs.

**Example:**
```python
# Agent previously used rm and lost important files
# Stored as antipattern: "Use trash instead of rm for safety"

# Later session, agent tries:
os.system("rm important_file.txt")

# Memory warning appears:
# ⚠️ ANTIPATTERN: Use trash instead of rm for safety
# Related to: file deletion (similarity: 0.92)
```

→ [Learn more about antipatterns](concepts/antipatterns.md)

### 🧠 Spreading Activation

When you recall one memory, activation flows through connected memories like neural pathways, surfacing related knowledge you didn't search for.

**Example:**
```python
recall(query="Etsy shop setup")

# Returns not just Etsy docs, but also:
# - "Browser automation on Etsy requires human-like pauses"
# - "Sign in with Apple bypasses bot detection"
# - "Printify API is better than browser automation"
```

→ [Learn more about spreading activation](concepts/spreading-activation.md)

### 🔗 Hebbian Learning

"Neurons that fire together wire together." When memories are recalled together, their connection strengthens automatically.

**Example:**
```python
# Agent recalls "Python" and "testing" in same session
# Connection "Python" ← testing → "pytest" strengthens

# Next time: recalling "Python" automatically surfaces "pytest"
```

→ [Learn more about Hebbian learning](concepts/hebbian-learning.md)

### 🧹 Memory Consolidation

Like sleep for your AI's brain. Periodically:
- Decays old, unused memories
- Merges duplicate knowledge
- Prunes weak connections
- Promotes important patterns

→ [Learn more about consolidation](concepts/consolidation.md)

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   AI Agent                       │
│         (Claude Code, Aider, etc.)               │
│                                                  │
│  Hooks: UserPromptSubmit → Recall context       │
│         PostToolUse → Learn from errors          │
│         Stop → Hebbian learning                  │
│                                                  │
│  MCP Tools: remember, recall, connect,           │
│             forget, amend, reflect, etc.          │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              avisual memories                     │
│                                                  │
│  Brain ──→ Retrieval (spreading activation)      │
│        ──→ Learning (Hebbian, auto-linking)       │
│        ──→ Consolidation (decay, merge, prune)   │
│        ──→ Context (budget compression)          │
│                                                  │
│  Storage: SQLite + sqlite-vec + FTS5             │
│  Embeddings: Ollama (nomic-embed-text)           │
└─────────────────────────────────────────────────┘
```

## Performance

- **602 atoms/sec** creation rate
- **1,128 synapses/sec** connection rate
- **0.7ms** average search query
- **1,545 recalls/sec** retrieval throughput

Tested with 1,000-atom datasets. See [benchmarks](https://github.com/avisual/memories/tree/main/tests/load).

## Use Cases

- **Claude Code** - Give Claude persistent memory across sessions
- **Aider** - Remember codebase patterns and past mistakes
- **Custom AI agents** - Any agent using MCP protocol
- **Research assistants** - Build knowledge graphs over time
- **Debugging agents** - Learn from errors, prevent repeats

## Next Steps

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quick-start.md)
- [Antipatterns Deep Dive](concepts/antipatterns.md)
- [API Reference](api/mcp-tools.md)
- [Best Practices](guides/best-practices.md)

## Support

- **GitHub Issues**: [github.com/avisual/memories/issues](https://github.com/avisual/memories/issues)
- **Documentation**: [avisual.github.io/memories](https://avisual.github.io/memories)
- **PyPI**: [pypi.org/project/avisual-memories](https://pypi.org/project/avisual-memories/)

## License

MIT - see [LICENSE](license.md) for details.
