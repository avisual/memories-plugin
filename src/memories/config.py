"""Central configuration for the memories system.

All tunables live here with sensible defaults.  Values can be overridden
through environment variables prefixed with ``MEMORIES_`` (nested keys use
double underscores, e.g. ``MEMORIES_RETRIEVAL__SEED_COUNT=20``).

Usage::

    from memories.config import get_config

    cfg = get_config()
    print(cfg.embedding_model)
    print(cfg.retrieval.seed_count)
"""

from __future__ import annotations

import os
import sys
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import TypeVar, get_type_hints

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Nested configuration sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SynapseTypeWeights:
    """Multipliers for different synapse types during spreading activation.

    Generic 'related-to' connections have reduced weight compared to
    semantically meaningful typed connections. This prevents embedding
    similarity from dominating over explicit human-labeled relationships.
    """

    caused_by: float = 1.0      # Direct causation - strongest signal
    warns_against: float = 1.0  # Safety warnings - always surface
    elaborates: float = 0.9     # Adds detail - high value
    supersedes: float = 0.8     # Newer version - relevant but context-dependent
    contradicts: float = 0.7    # Conflicting info - important but needs care
    part_of: float = 0.7        # Compositional - useful for navigation
    related_to: float = 0.4     # Generic embedding similarity - heavily discounted
    encoded_with: float = 0.2   # Contextual binding - weak but enables context-dependent recall


@dataclass(frozen=True, slots=True)
class RetrievalWeights:
    """Relative weights for the multi-signal scoring function.

    All weights should sum to approximately 1.0 for normalised scoring.
    """

    vector_similarity: float = 0.30
    spread_activation: float = 0.25
    recency: float = 0.08
    confidence: float = 0.12
    frequency: float = 0.02
    importance: float = 0.11
    bm25: float = 0.12


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Parameters that govern memory retrieval and activation spreading."""

    seed_count: int = 10
    spread_depth: int = 2
    decay_factor: float = 0.85
    min_activation: float = 0.1
    min_score: float = 0.25  # filter atoms below this composite score before budget fitting
    weights: RetrievalWeights = field(default_factory=RetrievalWeights)
    synapse_type_weights: SynapseTypeWeights = field(default_factory=SynapseTypeWeights)


@dataclass(frozen=True, slots=True)
class LearningConfig:
    """Parameters for automatic link formation and Hebbian strengthening."""

    auto_link_threshold: float = 0.82  # Raised to reduce generic hub-forming connections
    hebbian_increment: float = 0.05
    co_activation_window: int = 3
    min_accesses_for_hebbian: int = 1  # Atoms need at least this many accesses before Hebbian linking
    temporal_window_seconds: int = 300
    """Seconds within which two co-activated atoms are considered temporally proximate.

    Pairs accessed within this window receive the full Hebbian increment;
    pairs accessed further apart receive ``increment * 0.5``."""

    interference_confidence_penalty: float = 0.1
    """Confidence reduction applied to an older atom when a new atom is
    detected as contradicting it (retroactive interference).

    This implements the neuroscience principle that new competing memories
    weaken older conflicting ones immediately upon detection, rather than
    waiting for consolidation-time contradiction resolution."""

    max_new_pairs_per_session: int = 50
    """Maximum number of new synapses created per Hebbian session update.

    Large sessions (30+ atoms) generate O(n^2) candidate pairs.  Capping
    new synapse creation prevents the *fan effect* (Anderson 1974) where
    too many weak associations dilute the learning signal.  Pairs are
    prioritised by temporal proximity when timestamps are available."""

    stc_tagged_strength: float = 0.25
    """Initial strength for new auto-linked synapses under Synaptic Tagging
    and Capture.  New synapses start weak ("tagged") and must be reinforced
    by Hebbian co-activation within ``stc_capture_window_days`` to survive.
    Implements Frey & Morris (1997) synaptic tagging and capture."""

    stc_capture_window_days: int = 14
    """Days within which a tagged synapse must be Hebbian-reinforced to
    survive.  Tags that expire without reinforcement are deleted during
    consolidation.  Reinforced tags have their tag cleared (captured)."""


@dataclass(frozen=True, slots=True)
class ConsolidationConfig:
    """Parameters for memory decay, pruning, merging and compression."""

    decay_after_days: int = 30
    decay_rate: float = 0.95
    prune_threshold: float = 0.05
    merge_threshold: float = 0.95
    promote_access_count: int = 20
    compress_after_days: int = 60
    ltd_window_days: int = 14
    """Days without Hebbian co-activation before a synapse qualifies for LTD.

    A synapse between two individually-active atoms that hasn't been
    reinforced within this window is weakened by ``ltd_amount`` per
    consolidation cycle — implementing the anti-Hebbian rule: atoms that
    consistently fire apart lose their connection over time.
    """
    ltd_amount: float = 0.05
    """Deprecated — superseded by `ltd_fraction`. Will be removed in a future release."""
    ltd_fraction: float = 0.15
    """Proportional LTD multiplier: strength *= (1 - ltd_fraction) each cycle.
    Replaces the fixed ``ltd_amount`` subtraction so high-strength synapses
    lose more absolute strength than weak ones (naturally convergent)."""
    ltd_min_floor: float = 0.008
    """Absolute floor applied after proportional weakening — prevents immortal
    synapses that would asymptotically approach zero but never be pruned."""
    feedback_good_increment: float = 0.02
    """Importance nudge per unprocessed 'good' feedback signal."""
    feedback_bad_decrement: float = 0.03
    """Importance reduction per unprocessed 'bad' feedback signal (bad weighted 1.5×)."""
    abstraction_similarity: float = 0.82
    """Minimum similarity for experiences to count as the same cluster.

    Sits between ``auto_link_threshold`` (0.75) and ``merge_threshold``
    (0.95) — experiences in a cluster are clearly related but not
    near-identical copies.
    """
    abstraction_min_cluster: int = 5
    """Minimum number of corroborating experiences needed to emit a fact."""
    abstraction_min_age_days: int = 7
    """Experiences younger than this are not yet eligible for abstraction."""
    abstraction_max_per_cycle: int = 5
    """Cap on new facts created per consolidation run (keeps cycles fast)."""
    contradiction_resolution_threshold: float = 2.0
    """Ratio by which the winner's score must exceed the loser's before the
    contradiction is resolved.  Higher = more conservative (fewer resolutions)."""
    contradiction_resolution_decay: float = 0.15
    """Confidence decrement applied to the weaker atom when a contradiction
    is resolved in favour of the stronger one."""
    contradiction_min_age_days: int = 14
    """Both atoms must be at least this old before automatic resolution fires.
    Prevents resolving contradictions that were just created this session."""
    weight_tuning_learning_rate: float = 0.002
    """Step size for each per-signal weight adjustment during auto-tuning."""
    weight_tuning_max_drift: float = 0.30
    """Maximum fractional deviation (±) from factory defaults allowed for any weight."""
    weight_tuning_min_samples: int = 5
    """Minimum number of good AND bad feedback atoms required to run a tuning cycle."""
    dormant_cutoff_days: int = 30
    """Synapses that have never been activated and are older than this many
    days qualify for accelerated dormant decay."""
    dormant_multiplier: float = 0.80
    """Multiplicative decay factor applied per consolidation cycle to dormant
    (never-activated) synapses that exceed ``dormant_cutoff_days`` age."""
    type_decay_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "fact": 0.995,
            "skill": 0.995,
            "antipattern": 0.99,
            "preference": 0.97,
            "insight": 0.97,
            "experience": 0.93,
            "task": 1.0,
        }
    )
    """Per-type multiplier applied on top of the base decay_rate.
    Skills and facts decay slowly (long-lived knowledge); experiences decay
    faster (episodic, likely superseded by later experiences)."""

    hybrid_decay_transition_days: int = 90
    """Days of staleness after which decay transitions from exponential to
    power-law.  Atoms stale for fewer days than this use pure exponential
    decay; beyond this they transition to a slower power-law curve.

    Based on Wixted & Ebbesen (1991): short-term forgetting is exponential,
    long-term forgetting follows a power law (heavy-tail preservation)."""

    hybrid_decay_power_exponent: float = 0.5
    """Power-law exponent for the long-term decay phase.  Lower values
    produce slower long-term decay (heavier tail).  At the transition
    point the two curves are continuous."""

    type_feedback_inertia: dict[str, float] = field(
        default_factory=lambda: {
            "fact": 0.85,
            "skill": 0.85,
            "antipattern": 0.40,
            "preference": 0.50,
            "insight": 0.60,
            "experience": 0.30,
            "task": 0.20,
        }
    )
    """Per-type inertia for feedback-driven importance adjustments.

    Higher values mean the type resists feedback changes more.  Facts and
    skills are well-established knowledge that shouldn't flip easily;
    experiences and tasks are ephemeral and should respond quickly.
    Applied as: ``delta *= (1.0 - inertia)``."""

    ltp_tiers: dict[int, float] = field(
        default_factory=lambda: {
            5: 0.5,
            10: 0.33,
            20: 0.1,
        }
    )
    """Multi-scale long-term potentiation tiers.

    Maps access_count thresholds to decay protection factors.  An atom
    with access_count >= threshold gets its decay exponent multiplied by
    the factor — lower factors mean stronger protection.

    Example: an atom accessed 25 times qualifies for the ``20: 0.1`` tier,
    reducing its effective decay exponent to 10% (near-immunity)."""


@dataclass(frozen=True, slots=True)
class RateLimitConfig:
    """Rate limits for the Ollama embedding backend."""

    ollama_max_rps: int = 10
    ollama_batch_size: int = 32
    concurrent_embeds: int = 3


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MemoriesConfig:
    """Root configuration object for the memories system.

    All paths are stored as resolved :class:`~pathlib.Path` instances with
    ``~`` expanded.
    """

    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dims: int = 768
    db_path: Path = field(default_factory=lambda: Path("~/.memories/memories.db"))
    backup_dir: Path = field(default_factory=lambda: Path("~/.memories/backups"))
    backup_count: int = 5
    context_window_tokens: int = 200_000
    hook_budget_pct: float = 0.02  # 2% of context window for hook injection

    dedup_threshold: float = 0.92
    region_diversity_cap: int = 2

    distill_thinking: bool = False       # use a local LLM to distil reasoning atoms
    distill_model: str = "llama3.2:3b"  # Ollama model for distillation (generation)

    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    rate_limits: RateLimitConfig = field(default_factory=RateLimitConfig)

    def __post_init__(self) -> None:
        # Expand ~ in path fields.  We use object.__setattr__ because the
        # dataclass is frozen.
        object.__setattr__(self, "db_path", self.db_path.expanduser())
        object.__setattr__(self, "backup_dir", self.backup_dir.expanduser())


# ---------------------------------------------------------------------------
# Environment-variable loader
# ---------------------------------------------------------------------------

_ENV_PREFIX = "MEMORIES_"
_NESTED_SEP = "__"


def _resolve_type_hints(dc_type: type) -> dict[str, type]:
    """Resolve stringified annotations back to real types.

    ``from __future__ import annotations`` turns all annotations into
    strings.  :func:`typing.get_type_hints` evaluates them in the correct
    module namespace so we get the actual :class:`type` objects.
    """
    module = sys.modules.get(dc_type.__module__, None)
    globalns = getattr(module, "__dict__", {}) if module else {}
    return get_type_hints(dc_type, globalns=globalns)


def _coerce(value: str, target_type: type[T]) -> T:
    """Cast an env-var string to the target field type."""
    if target_type is bool:
        return target_type(value.lower() in ("1", "true", "yes"))  # type: ignore[return-value]
    if target_type is Path:
        return target_type(value)  # type: ignore[return-value]
    return target_type(value)  # type: ignore[return-value]


def _load_dataclass(dc_type: type[T], prefix: str) -> T:
    """Recursively build a dataclass from env-var overrides + defaults."""
    hints = _resolve_type_hints(dc_type)
    kwargs: dict[str, object] = {}

    for f in fields(dc_type):  # type: ignore[arg-type]
        field_type = hints[f.name]
        nested_prefix = f"{prefix}{f.name}{_NESTED_SEP}".upper()

        # Determine if this field is itself a dataclass.
        if hasattr(field_type, "__dataclass_fields__"):
            kwargs[f.name] = _load_dataclass(field_type, nested_prefix)
        else:
            env_key = f"{prefix}{f.name}".upper()
            raw = os.environ.get(env_key)
            if raw is not None:
                kwargs[f.name] = _coerce(raw, field_type)

    return dc_type(**kwargs)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_cached_config: MemoriesConfig | None = None


def get_config(*, reload: bool = False) -> MemoriesConfig:
    """Return the current :class:`MemoriesConfig`.

    On the first call the config is built by merging defaults with any
    ``MEMORIES_*`` environment variables.  The result is cached for the
    lifetime of the process unless *reload* is ``True``.

    Parameters
    ----------
    reload:
        Force re-reading environment variables and rebuilding the config.
    """
    global _cached_config  # noqa: PLW0603
    if _cached_config is None or reload:
        _cached_config = _load_dataclass(MemoriesConfig, _ENV_PREFIX)
    return _cached_config
