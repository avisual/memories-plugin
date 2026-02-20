# Privacy Policy — memories

**Last updated: February 2026**

## What memories collects

The memories plugin stores **locally derived content only**. It extracts and stores:

- Summaries of your Claude Code sessions (facts, insights, experiences, antipatterns)
- Tool call outcomes from your local development environment
- Session metadata (timestamps, project names derived from directory names)

All data is stored in a **local SQLite database** at `~/.memories/memories.db` on your machine.

## What memories does NOT collect

- memories does not transmit any data to external servers
- memories does not send data to the plugin author or any third party
- memories does not collect personally identifiable information beyond what you explicitly store
- memories does not access, read, or transmit the contents of your files

## Embedding model

memories uses [Ollama](https://ollama.com) to generate vector embeddings locally on your machine. Embedding generation happens entirely locally — no content is sent to cloud embedding services.

## Data control

You have full control over your data:

- **View**: `sqlite3 ~/.memories/memories.db "SELECT content FROM atoms LIMIT 20;"`
- **Delete all**: `rm ~/.memories/memories.db`
- **Soft-delete individual atoms**: Use the `mcp__memories__forget` tool
- **Export**: Use the `/memories-prune` skill

## Third-party services

memories has no third-party integrations. The only network call is to your local Ollama instance at `http://localhost:11434`.

## Changes

This policy may be updated as the plugin evolves. Changes will be reflected in this file and in the repository changelog.

## Contact

For questions or to report a vulnerability: https://github.com/avisual/memories-plugin/issues
