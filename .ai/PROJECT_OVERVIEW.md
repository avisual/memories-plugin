# Memories - Brain-Like Long-Term Memory System for Claude Code

## Purpose
Revolutionary long-term memory system for Claude Code enabling persistent learning and context retention across sessions. Implements brain-inspired memory architecture with automatic consolidation, semantic retrieval, and adaptive importance scoring, transforming ephemeral AI conversations into continuous knowledge building.

## Tech Stack
- **Language**: TypeScript/JavaScript (9,725 LOC)
- **Storage**: File-based memory persistence (JSON/JSONL)
- **Integration**: Claude Code CLI skill system
- **Architecture**: Hippocampus-inspired memory consolidation
- **Search**: Semantic similarity and keyword-based retrieval

## Key Features

**Brain-Inspired Architecture**:
- Short-term memory (working context)
- Long-term memory (persistent storage)
- Memory consolidation pipeline (sleep-like processing)
- Importance-based retention (adaptive pruning)
- Semantic clustering (related memory grouping)

**Automatic Memory Capture**:
- Session activity tracking
- Key decision recording
- Error pattern learning
- Successful solution caching
- User preference retention

**Intelligent Retrieval**:
- Context-aware memory search
- Semantic similarity matching
- Temporal relevance scoring
- Frequency-based importance
- Multi-dimensional ranking

**Memory Management**:
- Automatic consolidation (merge similar memories)
- Adaptive forgetting (low-importance pruning)
- Memory compression (summarization)
- Cross-session continuity
- Memory health monitoring

## Architecture

```
Claude Code Session (Working Memory)
  ↓
Memory Capture System
  ├── Activity Monitoring
  ├── Importance Scoring
  └── Event Classification
      ↓
Short-Term Memory Buffer
  ↓ (Consolidation Trigger)
Hippocampus-Like Processor
  ├── Semantic Clustering
  ├── Duplicate Detection
  ├── Importance Re-ranking
  └── Compression Pipeline
      ↓
Long-Term Memory Store (Persistent)
  ├── Episodic Memories (events, sessions)
  ├── Semantic Memories (patterns, solutions)
  ├── Procedural Memories (workflows, commands)
  └── Meta Memories (preferences, context)
      ↓
Retrieval System (Context-Aware Search)
```

**Memory Types**:
1. **Episodic**: Specific events, debugging sessions, feature implementations
2. **Semantic**: General knowledge, patterns, best practices
3. **Procedural**: Workflows, command sequences, automation scripts
4. **Meta**: User preferences, project context, communication style

## Deployment

**Installation**: Claude Code skill (`/memories` or `/mem`)

**Configuration**:
```json
{
  "storage_path": "~/.claude/memories/",
  "consolidation_interval": "daily",
  "max_memories": 10000,
  "importance_threshold": 0.3,
  "semantic_similarity_threshold": 0.85
}
```

**Usage Commands**:
```
/mem save "Important insight about X"
/mem recall "How did we solve Y?"
/mem list --recent 10
/mem consolidate --force
/mem stats
```

## Integration Points
- **Claude Code CLI**: Native skill integration
- **Session Context**: Automatic capture from active conversations
- **File System**: Persistent storage in user's .claude directory
- **Semantic Engine**: Embedding-based similarity search
- **User Workflow**: Transparent background operation

## File Structure
```
memories/ [9,725 LOC TypeScript]
├── src/
│   ├── core/
│   │   ├── memory_capture.ts       # Activity monitoring
│   │   ├── consolidation.ts        # Hippocampus-like processing
│   │   ├── retrieval.ts            # Semantic search
│   │   └── importance_scorer.ts    # Adaptive ranking
│   ├── storage/
│   │   ├── memory_store.ts         # Persistent layer
│   │   ├── indexer.ts              # Fast lookup
│   │   └── compression.ts          # Memory optimization
│   ├── types/
│   │   ├── memory_types.ts         # Type definitions
│   │   └── schemas.ts              # Validation
│   └── cli/
│       ├── commands.ts             # User commands
│       └── formatters.ts           # Output rendering
├── tests/
│   ├── consolidation.test.ts
│   ├── retrieval.test.ts
│   └── integration.test.ts
├── package.json
├── tsconfig.json
└── README.md
```

## Use Cases
- Persistent learning across Claude Code sessions
- Automatic capture of debugging insights
- Project-specific context retention
- User preference learning
- Solution pattern caching
- Error prevention (remember past mistakes)
- Workflow automation (remember successful sequences)
- Cross-project knowledge transfer

## Development Status
- **Version**: Early prototype/experimental
- **Codebase**: 9,725 lines TypeScript
- **Status**: Active development
- **Integration**: Claude Code skill system
- **Architecture**: Brain-inspired memory consolidation
- **Testing**: Unit + integration test coverage

## Benefits

**For Users**:
- Claude "remembers" your projects and preferences
- Faster problem-solving (recall past solutions)
- Continuous improvement across sessions
- Reduced repetitive explanations
- Personalized assistance

**For Claude**:
- Long-term context accumulation
- Pattern recognition across projects
- Adaptive learning from user feedback
- Session continuity maintenance
- Knowledge compound growth

## Technical Innovation

**Hippocampus-Inspired Consolidation**:
- Mimics human memory transfer from short-term to long-term
- Importance-based filtering (what to remember vs. forget)
- Semantic clustering (organize related memories)
- Compression (summarize verbose details)
- Adaptive forgetting (prune low-value memories)

**Retrieval Mechanisms**:
- Context-aware ranking (what's relevant NOW)
- Temporal decay modeling (recent = more relevant)
- Frequency-importance balance
- Multi-factor scoring (semantic + temporal + frequency)

## Future Potential
- Multi-user memory isolation
- Shared team memories
- Memory export/import
- Cross-agent memory sharing
- Advanced semantic embeddings
- Memory visualization dashboards
- Automatic insight extraction
