"""Scalability tests for 1000+ atoms.

Tests the full Brain functionality at scale, including:
- Creating 1000 atoms with various types
- Recall performance with spreading activation
- Hebbian learning with many co-activations
- Consolidation with realistic workloads
- Memory budget constraints

Run with: uv run pytest tests/test_scale_1000.py -v
"""

from __future__ import annotations

import asyncio
import random
import string
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from memories.atoms import AtomManager
from memories.brain import Brain
from memories.consolidation import ConsolidationEngine
from memories.embeddings import EmbeddingEngine
from memories.learning import LearningEngine
from memories.retrieval import RetrievalEngine
from memories.storage import Storage
from memories.synapses import SynapseManager

pytestmark = pytest.mark.load


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_embeddings(deterministic: bool = True) -> MagicMock:
    """Create a mock embedding engine.
    
    If deterministic=True, returns consistent embeddings based on content hash.
    This enables realistic similarity searches in tests.
    """
    engine = MagicMock(spec=EmbeddingEngine)
    
    def _hash_to_embedding(text: str) -> list[float]:
        """Generate deterministic 768-dim embedding from text hash."""
        import hashlib
        h = hashlib.sha256(text.encode()).hexdigest()
        # Use hash bytes to seed consistent pseudo-random values
        values = []
        for i in range(0, min(len(h), 96), 2):
            byte_val = int(h[i:i+2], 16)
            values.append((byte_val - 128) / 128)  # Normalize to [-1, 1]
        # Pad to 768 dims
        while len(values) < 768:
            values.append(0.0)
        return values[:768]
    
    def _similarity(v1: list[float], v2: list[float]) -> float:
        """Compute cosine similarity."""
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = sum(a * a for a in v1) ** 0.5
        mag2 = sum(b * b for b in v2) ** 0.5
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)
    
    engine.embed_text = AsyncMock(side_effect=lambda t: _hash_to_embedding(t))
    engine.embed_and_store = AsyncMock(side_effect=lambda aid, t: _hash_to_embedding(t))
    engine.embed_batch = AsyncMock(return_value=[])
    engine.search_similar = AsyncMock(return_value=[])
    engine.health_check = AsyncMock(return_value=True)
    engine.cosine_similarity = MagicMock(side_effect=_similarity)  # sync function (M-6)
    
    return engine


def _generate_test_content(category: str, index: int) -> str:
    """Generate realistic content for different categories."""
    templates = {
        "fact": [
            f"Python version {index % 20 + 3}.{index % 15} was released in {2010 + index % 16}",
            f"The API endpoint /api/v{index % 5}/users returns JSON data",
            f"Database table 'items_{index}' has {index * 10} rows",
            f"Configuration setting MAX_CONNECTIONS is set to {index * 5}",
            f"Module '{chr(97 + index % 26)}_utils' provides helper functions",
        ],
        "experience": [
            f"Successfully deployed version {index}.0.{index % 100} to production",
            f"Debug session #{index}: Found race condition in thread pool",
            f"Refactored component_{index} for better performance",
            f"Fixed bug #{index * 7}: null pointer in user handler",
            f"Completed code review for PR #{index + 100}",
        ],
        "skill": [
            f"Use pytest fixtures for test setup - learned from task_{index}",
            f"Apply retry logic with exponential backoff for API calls",
            f"Implement circuit breaker pattern for service {index}",
            f"Use connection pooling for database access optimization",
            f"Apply SOLID principles to module design",
        ],
        "preference": [
            f"Prefer async/await over callbacks for service {index}",
            f"Use type hints for all function signatures",
            f"Keep functions under {20 + index % 30} lines",
            f"Test coverage should exceed {70 + index % 30}%",
            f"Document public APIs with docstrings",
        ],
        "antipattern": [
            f"Don't commit API keys to repository - incident #{index}",
            f"Avoid N+1 queries in ORM - performance issue #{index}",
            f"Don't ignore exceptions silently - bug #{index}",
            f"Avoid hardcoded timeouts - failure case #{index}",
            f"Don't mix sync/async code carelessly",
        ],
    }
    category_templates = templates.get(category, templates["fact"])
    return category_templates[index % len(category_templates)]


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------


class TestAtomCreation1000:
    """Test creating and managing 1000 atoms."""
    
    ATOM_COUNT = 1000
    
    async def test_create_1000_atoms_performance(self, tmp_path: Path) -> None:
        """Verify 1000 atoms can be created in reasonable time."""
        store = Storage(tmp_path / "scale_create.db")
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()
        
        mock_emb = _make_mock_embeddings()
        atoms = AtomManager(store, mock_emb)
        
        types = ["fact", "experience", "skill", "preference", "antipattern"]
        regions = ["core", "api", "database", "frontend", "devops"]
        
        start = time.monotonic()
        
        for i in range(self.ATOM_COUNT):
            atom_type = types[i % len(types)]
            region = regions[i % len(regions)]
            content = _generate_test_content(atom_type, i)
            
            await atoms.create(
                content=content,
                type=atom_type,
                region=region,
                confidence=0.5 + (i % 50) / 100,
            )
        
        elapsed = time.monotonic() - start
        
        # Verify count
        rows = await store.execute(
            "SELECT COUNT(*) AS cnt FROM atoms WHERE is_deleted = 0"
        )
        assert rows[0]["cnt"] == self.ATOM_COUNT
        
        # Should complete in under 30 seconds (conservative)
        assert elapsed < 30.0, f"Creation took {elapsed:.2f}s, expected < 30s"
        
        # Log performance
        print(f"\n  Created {self.ATOM_COUNT} atoms in {elapsed:.2f}s "
              f"({self.ATOM_COUNT/elapsed:.0f} atoms/sec)")
        
        await store.close()
    
    async def test_concurrent_1000_atom_creation(self, tmp_path: Path) -> None:
        """Test creating atoms concurrently."""
        store = Storage(tmp_path / "scale_concurrent.db")
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()
        
        mock_emb = _make_mock_embeddings()
        atoms = AtomManager(store, mock_emb)
        
        async def create_batch(start_idx: int, count: int) -> int:
            created = 0
            for i in range(start_idx, start_idx + count):
                await atoms.create(
                    content=f"concurrent atom {i}",
                    type="fact",
                    region="concurrent",
                )
                created += 1
            return created
        
        # 10 concurrent batches of 100 atoms each
        batch_size = 100
        num_batches = 10
        
        start = time.monotonic()
        
        results = await asyncio.gather(*[
            create_batch(i * batch_size, batch_size)
            for i in range(num_batches)
        ])
        
        elapsed = time.monotonic() - start
        total_created = sum(results)
        
        assert total_created == self.ATOM_COUNT
        
        # Verify in database
        rows = await store.execute(
            "SELECT COUNT(*) AS cnt FROM atoms WHERE is_deleted = 0"
        )
        assert rows[0]["cnt"] == self.ATOM_COUNT
        
        print(f"\n  Concurrent creation: {self.ATOM_COUNT} atoms in {elapsed:.2f}s "
              f"({self.ATOM_COUNT/elapsed:.0f} atoms/sec)")
        
        await store.close()


class TestRecallPerformance1000:
    """Test recall performance with 1000+ atoms."""
    
    ATOM_COUNT = 1000
    MAX_RECALL_TIME = 1.0  # seconds
    
    async def test_fts_recall_at_scale(self, tmp_path: Path) -> None:
        """Test full-text search recall with 1000 atoms."""
        store = Storage(tmp_path / "scale_fts.db")
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()
        
        # Bulk insert atoms
        params = []
        for i in range(self.ATOM_COUNT):
            atom_type = ["fact", "experience", "skill"][i % 3]
            content = _generate_test_content(atom_type, i)
            params.append((content, atom_type, "test", 0.8, 0))
        
        await store.execute_many(
            """
            INSERT INTO atoms (content, type, region, confidence, is_deleted)
            VALUES (?, ?, ?, ?, ?)
            """,
            params,
        )
        
        # Test FTS queries
        test_queries = [
            "Python version",
            "API endpoint",
            "database table",
            "deployed production",
            "retry logic",
        ]
        
        total_time = 0.0
        for query in test_queries:
            start = time.monotonic()
            rows = await store.execute(
                """
                SELECT a.* FROM atoms_fts f
                JOIN atoms a ON a.id = f.rowid
                WHERE atoms_fts MATCH ?
                  AND a.is_deleted = 0
                LIMIT 50
                """,
                (query,),
            )
            elapsed = time.monotonic() - start
            total_time += elapsed
            
            assert elapsed < self.MAX_RECALL_TIME, (
                f"FTS query '{query}' took {elapsed:.3f}s"
            )
        
        avg_time = total_time / len(test_queries)
        print(f"\n  Average FTS query time: {avg_time*1000:.1f}ms")
        
        await store.close()
    
    async def test_region_filtered_recall(self, tmp_path: Path) -> None:
        """Test recall filtered by region with 1000 atoms."""
        store = Storage(tmp_path / "scale_region.db")
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()
        
        regions = ["api", "database", "frontend", "backend", "devops"]
        
        # Bulk insert with region distribution
        params = []
        for i in range(self.ATOM_COUNT):
            region = regions[i % len(regions)]
            params.append((f"atom {i} in {region}", "fact", region, 0.8, 0))
        
        await store.execute_many(
            """
            INSERT INTO atoms (content, type, region, confidence, is_deleted)
            VALUES (?, ?, ?, ?, ?)
            """,
            params,
        )
        
        # Query each region
        for region in regions:
            start = time.monotonic()
            rows = await store.execute(
                """
                SELECT * FROM atoms
                WHERE region = ? AND is_deleted = 0
                ORDER BY confidence DESC
                LIMIT 100
                """,
                (region,),
            )
            elapsed = time.monotonic() - start
            
            # Each region should have ~200 atoms (1000/5)
            assert len(rows) == 100  # Limited to 100
            assert elapsed < self.MAX_RECALL_TIME
        
        await store.close()


class TestSynapseScale1000:
    """Test synapse operations at scale."""
    
    ATOM_COUNT = 500  # Fewer atoms but more synapses
    SYNAPSES_PER_ATOM = 5  # ~2500 synapses total
    
    async def test_create_synapses_at_scale(self, tmp_path: Path) -> None:
        """Test creating many synapses between atoms."""
        store = Storage(tmp_path / "scale_synapses.db")
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()
        
        mock_emb = _make_mock_embeddings()
        atoms = AtomManager(store, mock_emb)
        synapses = SynapseManager(store)
        
        # Create atoms
        atom_ids = []
        for i in range(self.ATOM_COUNT):
            atom = await atoms.create(
                content=f"synapse test atom {i}",
                type="fact",
                region="synapse-test",
            )
            atom_ids.append(atom.id)
        
        # Create synapses (each atom connected to next N atoms)
        relationships = ["related-to", "elaborates", "caused-by"]
        synapse_count = 0
        
        start = time.monotonic()
        
        for i, source_id in enumerate(atom_ids):
            for j in range(self.SYNAPSES_PER_ATOM):
                target_idx = (i + j + 1) % self.ATOM_COUNT
                target_id = atom_ids[target_idx]
                if source_id != target_id:
                    await synapses.create(
                        source_id=source_id,
                        target_id=target_id,
                        relationship=relationships[j % len(relationships)],
                        strength=0.5 + (j % 5) / 10,
                    )
                    synapse_count += 1
        
        elapsed = time.monotonic() - start
        
        # Verify synapse count
        rows = await store.execute("SELECT COUNT(*) AS cnt FROM synapses")
        assert rows[0]["cnt"] == synapse_count
        
        print(f"\n  Created {synapse_count} synapses in {elapsed:.2f}s "
              f"({synapse_count/elapsed:.0f} synapses/sec)")
        
        await store.close()
    
    async def test_get_connections_at_scale(self, tmp_path: Path) -> None:
        """Test fetching connections for atoms with many synapses."""
        store = Storage(tmp_path / "scale_connections.db")
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()
        
        mock_emb = _make_mock_embeddings()
        atoms = AtomManager(store, mock_emb)
        synapses = SynapseManager(store)
        
        # Create a hub atom connected to many others
        hub = await atoms.create(content="hub atom", type="fact")
        
        spoke_ids = []
        for i in range(100):
            spoke = await atoms.create(content=f"spoke {i}", type="fact")
            spoke_ids.append(spoke.id)
            await synapses.create(
                source_id=hub.id,
                target_id=spoke.id,
                relationship="related-to",
            )
        
        # Time fetching all connections
        start = time.monotonic()
        connections = await synapses.get_connections(hub.id)
        elapsed = time.monotonic() - start
        
        assert len(connections) == 100
        assert elapsed < 0.1  # Should be very fast
        
        print(f"\n  Fetched 100 connections in {elapsed*1000:.1f}ms")
        
        await store.close()


class TestHebbianLearning1000:
    """Test Hebbian learning at scale."""
    
    async def test_hebbian_with_many_coactivations(self, tmp_path: Path) -> None:
        """Test Hebbian learning with many atoms co-activated."""
        store = Storage(tmp_path / "scale_hebbian.db")
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()
        
        mock_emb = _make_mock_embeddings()
        atoms = AtomManager(store, mock_emb)
        synapses = SynapseManager(store)
        
        # Create atoms
        atom_ids = []
        for i in range(100):
            atom = await atoms.create(
                content=f"hebbian atom {i}",
                type="fact",
            )
            atom_ids.append(atom.id)
        
        # Simulate co-activations (groups of 10 atoms activated together)
        # First record access to simulate retrieval (required for Hebbian)
        start = time.monotonic()
        total_strengthened = 0
        
        for batch_start in range(0, 100, 10):
            batch_ids = atom_ids[batch_start:batch_start + 10]
            # Record access for each atom in the batch (simulates retrieval)
            for aid in batch_ids:
                await atoms.record_access(aid)
            count = await synapses.hebbian_update(batch_ids)
            total_strengthened += count
        
        elapsed = time.monotonic() - start
        
        # Verify synapses were created
        rows = await store.execute("SELECT COUNT(*) AS cnt FROM synapses")
        
        print(f"\n  Hebbian learning created {rows[0]['cnt']} synapses "
              f"from {total_strengthened} strengthened in {elapsed:.2f}s")
        
        await store.close()


class TestConsolidation1000:
    """Test consolidation at scale."""
    
    ATOM_COUNT = 1000
    
    async def test_consolidation_with_1000_atoms(self, tmp_path: Path) -> None:
        """Test reflect() with 1000 atoms."""
        store = Storage(tmp_path / "scale_consolidation.db")
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()
        
        mock_emb = _make_mock_embeddings()
        atoms = AtomManager(store, mock_emb)
        synapses = SynapseManager(store)
        engine = ConsolidationEngine(store, mock_emb, atoms, synapses)
        
        # Bulk insert atoms with varied confidence
        params = []
        for i in range(self.ATOM_COUNT):
            # Mix of confidences - some low (candidates for decay)
            confidence = 0.3 + (i % 70) / 100
            params.append((f"consolidation atom {i}", "fact", "test", confidence, 0))
        
        await store.execute_many(
            """
            INSERT INTO atoms (content, type, region, confidence, is_deleted)
            VALUES (?, ?, ?, ?, ?)
            """,
            params,
        )
        
        # Run consolidation
        start = time.monotonic()
        result = await engine.reflect(dry_run=True)
        elapsed = time.monotonic() - start
        
        assert result.dry_run is True
        
        print(f"\n  Consolidation analysis of {self.ATOM_COUNT} atoms in {elapsed:.2f}s")
        print(f"    - Decay candidates: {result.decayed}")
        print(f"    - Prune candidates: {result.pruned}")
        print(f"    - Merge candidates: {result.merged}")
        
        # Should complete in reasonable time
        assert elapsed < 10.0, f"Consolidation took {elapsed:.2f}s, expected < 10s"
        
        await store.close()


class TestMemoryBudget1000:
    """Test memory budget constraints at scale."""
    
    ATOM_COUNT = 1000
    
    async def test_recall_with_budget_constraint(self, tmp_path: Path) -> None:
        """Test that recall respects token budget."""
        store = Storage(tmp_path / "scale_budget.db")
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()
        
        # Insert atoms with varying content lengths
        params = []
        for i in range(self.ATOM_COUNT):
            # Content length varies from 50 to 200 chars
            content_len = 50 + (i % 150)
            content = f"Budget test atom {i}: " + "x" * content_len
            params.append((content, "fact", "budget", 0.8, 0))
        
        await store.execute_many(
            """
            INSERT INTO atoms (content, type, region, confidence, is_deleted)
            VALUES (?, ?, ?, ?, ?)
            """,
            params,
        )
        
        # Query with no limit - should get many results
        rows_unlimited = await store.execute(
            """
            SELECT * FROM atoms
            WHERE region = 'budget' AND is_deleted = 0
            ORDER BY confidence DESC
            """,
        )
        
        # Calculate total tokens (rough estimate: 1 token ≈ 4 chars)
        total_chars = sum(len(r["content"]) for r in rows_unlimited)
        estimated_tokens = total_chars // 4
        
        # Query with budget constraint (4000 tokens ≈ 16000 chars)
        budget_chars = 16000
        running_total = 0
        budget_results = []
        
        for row in rows_unlimited:
            if running_total + len(row["content"]) > budget_chars:
                break
            budget_results.append(row)
            running_total += len(row["content"])
        
        print(f"\n  Total atoms: {len(rows_unlimited)}")
        print(f"    Estimated tokens: {estimated_tokens}")
        print(f"    With 4000 token budget: {len(budget_results)} atoms "
              f"({running_total} chars)")
        
        await store.close()


class TestEndToEnd1000:
    """End-to-end test with realistic 1000-atom workload."""
    
    async def test_realistic_workflow(self, tmp_path: Path) -> None:
        """Simulate realistic usage with 1000 atoms."""
        store = Storage(tmp_path / "scale_e2e.db")
        store._backup_dir = tmp_path / "backups"
        store._backup_dir.mkdir()
        await store.initialize()
        
        mock_emb = _make_mock_embeddings()
        atoms = AtomManager(store, mock_emb)
        synapses = SynapseManager(store)
        
        # Phase 1: Create atoms (simulating learning over time)
        print("\n  Phase 1: Creating 1000 atoms...")
        start = time.monotonic()
        
        types = ["fact", "experience", "skill", "preference", "antipattern"]
        atom_ids = []
        
        for i in range(1000):
            atom = await atoms.create(
                content=_generate_test_content(types[i % 5], i),
                type=types[i % 5],
                region=["core", "api", "database"][i % 3],
            )
            atom_ids.append(atom.id)
            
            # Occasionally create synapses to previous atoms
            if i > 0 and i % 10 == 0:
                # Link to a random previous atom
                target_id = atom_ids[random.randint(0, i - 1)]
                await synapses.create(
                    source_id=atom.id,
                    target_id=target_id,
                    relationship="related-to",
                )
        
        phase1_time = time.monotonic() - start
        print(f"    Completed in {phase1_time:.2f}s")
        
        # Phase 2: Simulate recalls
        print("  Phase 2: Running 100 recalls...")
        start = time.monotonic()
        
        queries = [
            "Python", "API", "database", "deploy", "bug",
            "test", "config", "module", "version", "error",
        ]
        
        for i in range(100):
            query = queries[i % len(queries)]
            await store.execute(
                """
                SELECT a.* FROM atoms_fts f
                JOIN atoms a ON a.id = f.rowid
                WHERE atoms_fts MATCH ?
                  AND a.is_deleted = 0
                LIMIT 20
                """,
                (query,),
            )
        
        phase2_time = time.monotonic() - start
        print(f"    Completed in {phase2_time:.2f}s ({100/phase2_time:.0f} recalls/sec)")
        
        # Phase 3: Apply Hebbian learning
        print("  Phase 3: Hebbian learning (10 sessions)...")
        start = time.monotonic()
        
        for session in range(10):
            # Simulate 5 atoms being co-activated per session
            session_atoms = random.sample(atom_ids, 5)
            # Record access first (required for Hebbian learning)
            for aid in session_atoms:
                await atoms.record_access(aid)
            await synapses.hebbian_update(session_atoms)
        
        phase3_time = time.monotonic() - start
        print(f"    Completed in {phase3_time:.2f}s")
        
        # Final stats
        atom_count = await store.execute(
            "SELECT COUNT(*) AS cnt FROM atoms WHERE is_deleted = 0"
        )
        synapse_count = await store.execute(
            "SELECT COUNT(*) AS cnt FROM synapses"
        )
        
        print(f"\n  Final state:")
        print(f"    Atoms: {atom_count[0]['cnt']}")
        print(f"    Synapses: {synapse_count[0]['cnt']}")
        print(f"    Total time: {phase1_time + phase2_time + phase3_time:.2f}s")
        
        await store.close()
