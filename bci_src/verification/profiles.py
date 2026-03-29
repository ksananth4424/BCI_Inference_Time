"""
Verifier Profile Management System.

Immutable, version-controlled verifier configurations for deterministic claim verification.
Each profile defines:
  - Similarity/tolerance thresholds
  - Synonym mappings (colors, sizes, actions)
  - Contradiction rules (what claims directly contradict each other)
  - Confidence parameters

Profiles are frozen dataclasses to ensure reproducibility across runs.
"""

from dataclasses import dataclass
from typing import Dict, Set, Tuple, Optional
import json
from pathlib import Path


@dataclass(frozen=True)
class VerifierProfile:
    """
    Immutable verifier configuration. Version-controlled for reproducibility.
    
    Attributes:
        name: Profile identifier (e.g., "strict", "balanced", "high_recall")
        version: Semantic version of this profile (e.g., "1.0")
        description: Human-readable explanation of profile purpose
        
        # Thresholds
        ATTRIBUTE_SIMILARITY_THRESHOLD: Fuzzy match score for attribute verification (0.0-1.0)
            Higher = stricter (fewer false positives, more false negatives)
        SPATIAL_TOLERANCE: IoU tolerance for spatial relation verification (0.0-1.0)
            Higher = more permissive (more false positives)
        OBJECT_PRESENCE_CONFIDENCE: Min confidence for object existence (0.0-1.0)
            Higher = only high-confidence contradictions count
        
        # Mappings
        COLOR_SYNONYMS: Colors that are considered equivalent (e.g., {"red": {"crimson", "scarlet"}})
        SIZE_SYNONYMS: Sizes that are considered equivalent
        ACTION_SYNONYMS: Actions that are considered equivalent
        
        # Contradictions
        COLOR_CONTRADICTIONS: Colors that directly contradict (e.g., {"red": {"blue", "green"}})
        SIZE_CONTRADICTIONS: Sizes that directly contradict
    """
    
    name: str
    version: str
    description: str
    
    # Thresholds (0.0-1.0 range)
    ATTRIBUTE_SIMILARITY_THRESHOLD: float
    SPATIAL_TOLERANCE: float
    OBJECT_PRESENCE_CONFIDENCE: float
    
    # Synonym mappings
    COLOR_SYNONYMS: Dict[str, Set[str]]
    SIZE_SYNONYMS: Dict[str, Set[str]]
    ACTION_SYNONYMS: Dict[str, Set[str]]
    
    # Contradiction rules
    COLOR_CONTRADICTIONS: Dict[str, Set[str]]
    SIZE_CONTRADICTIONS: Dict[str, Set[str]]
    
    def __post_init__(self):
        """Validate profile after initialization."""
        if not 0.0 <= self.ATTRIBUTE_SIMILARITY_THRESHOLD <= 1.0:
            raise ValueError(
                f"ATTRIBUTE_SIMILARITY_THRESHOLD must be in [0.0, 1.0], "
                f"got {self.ATTRIBUTE_SIMILARITY_THRESHOLD}"
            )
        if not 0.0 <= self.SPATIAL_TOLERANCE <= 1.0:
            raise ValueError(
                f"SPATIAL_TOLERANCE must be in [0.0, 1.0], "
                f"got {self.SPATIAL_TOLERANCE}"
            )
        if not 0.0 <= self.OBJECT_PRESENCE_CONFIDENCE <= 1.0:
            raise ValueError(
                f"OBJECT_PRESENCE_CONFIDENCE must be in [0.0, 1.0], "
                f"got {self.OBJECT_PRESENCE_CONFIDENCE}"
            )
    
    def to_dict(self) -> Dict:
        """Serialize profile to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "ATTRIBUTE_SIMILARITY_THRESHOLD": self.ATTRIBUTE_SIMILARITY_THRESHOLD,
            "SPATIAL_TOLERANCE": self.SPATIAL_TOLERANCE,
            "OBJECT_PRESENCE_CONFIDENCE": self.OBJECT_PRESENCE_CONFIDENCE,
            "COLOR_SYNONYMS": {k: list(v) for k, v in self.COLOR_SYNONYMS.items()},
            "SIZE_SYNONYMS": {k: list(v) for k, v in self.SIZE_SYNONYMS.items()},
            "ACTION_SYNONYMS": {k: list(v) for k, v in self.ACTION_SYNONYMS.items()},
            "COLOR_CONTRADICTIONS": {k: list(v) for k, v in self.COLOR_CONTRADICTIONS.items()},
            "SIZE_CONTRADICTIONS": {k: list(v) for k, v in self.SIZE_CONTRADICTIONS.items()},
        }
    
    def to_json(self, path: Optional[Path] = None) -> Optional[str]:
        """Serialize to JSON string or file."""
        data = self.to_dict()
        if path:
            Path(path).write_text(json.dumps(data, indent=2))
            return None
        return json.dumps(data, indent=2)


# =============================================================================
# PREDEFINED PROFILES (Version 1.0)
# =============================================================================

STRICT_PROFILE = VerifierProfile(
    name="strict",
    version="1.0",
    description=(
        "Conservative verifier: high precision, lower recall. "
        "Use for trust-critical experiments. Requires high confidence before marking contradictions."
    ),
    ATTRIBUTE_SIMILARITY_THRESHOLD=0.90,
    SPATIAL_TOLERANCE=0.10,
    OBJECT_PRESENCE_CONFIDENCE=0.95,
    COLOR_SYNONYMS={
        "red": {"crimson", "scarlet", "maroon"},
        "blue": {"navy", "azure", "cobalt"},
        "green": {"lime", "olive", "sage"},
        "yellow": {"golden", "amber"},
        "black": {"charcoal", "ebony"},
        "white": {"ivory", "cream", "off-white"},
        "gray": {"grey", "silver", "ash"},
        "orange": {"coral", "burnt-orange", "peach"},
        "purple": {"violet", "plum", "magenta"},
        "brown": {"tan", "beige", "khaki"},
    },
    SIZE_SYNONYMS={
        "small": {"tiny", "little", "petite"},
        "large": {"big", "huge", "massive"},
        "medium": {"mid-sized", "moderate"},
    },
    ACTION_SYNONYMS={
        "sitting": {"sits", "seated"},
        "standing": {"stands"},
        "walking": {"walks", "strolling"},
        "running": {"runs", "sprinting"},
        "lying": {"lays", "reclining"},
    },
    COLOR_CONTRADICTIONS={
        "red": {"blue", "green", "yellow", "purple", "white", "black"},
        "blue": {"red", "yellow", "orange", "brown", "black"},
        "green": {"red", "pink", "purple", "orange"},
        "yellow": {"blue", "purple", "black"},
        "black": {"white", "yellow", "red"},
        "white": {"black", "red", "blue", "green", "purple"},
    },
    SIZE_CONTRADICTIONS={
        "small": {"large", "huge", "massive"},
        "large": {"small", "tiny", "little"},
    },
)

BALANCED_PROFILE = VerifierProfile(
    name="balanced",
    version="1.0",
    description=(
        "Balanced verifier (DEFAULT): moderate precision/recall. "
        "Recommended for main experiments. Good tradeoff for causal analysis."
    ),
    ATTRIBUTE_SIMILARITY_THRESHOLD=0.80,
    SPATIAL_TOLERANCE=0.15,
    OBJECT_PRESENCE_CONFIDENCE=0.85,
    COLOR_SYNONYMS={
        "red": {"crimson", "scarlet", "maroon", "pink", "rose"},
        "blue": {"navy", "azure", "cobalt", "cyan", "teal"},
        "green": {"lime", "olive", "sage", "forest", "mint"},
        "yellow": {"golden", "amber", "gold", "lemon"},
        "black": {"charcoal", "ebony", "dark"},
        "white": {"ivory", "cream", "off-white", "pearl"},
        "gray": {"grey", "silver", "ash", "slate"},
        "orange": {"coral", "burnt-orange", "peach", "apricot"},
        "purple": {"violet", "plum", "magenta", "lavender"},
        "brown": {"tan", "beige", "khaki", "chocolate", "caramel"},
    },
    SIZE_SYNONYMS={
        "small": {"tiny", "little", "petite", "compact"},
        "large": {"big", "huge", "massive", "gigantic"},
        "medium": {"mid-sized", "moderate", "standard"},
    },
    ACTION_SYNONYMS={
        "sit": {"sitting", "sits", "seated"},
        "stand": {"standing", "stands", "upright"},
        "walk": {"walking", "walks", "strolling", "pacing"},
        "run": {"running", "runs", "sprinting", "jogging"},
        "lie": {"lying", "lays", "reclining", "sleeping"},
        "jump": {"jumping", "jumps", "leaping"},
        "play": {"playing", "plays"},
    },
    COLOR_CONTRADICTIONS={
        "red": {"blue", "green", "yellow", "purple", "white", "black", "cyan"},
        "blue": {"red", "yellow", "orange", "brown", "black", "green"},
        "green": {"red", "pink", "purple", "orange", "blue"},
        "yellow": {"blue", "purple", "black"},
        "black": {"white", "yellow", "red", "purple"},
        "white": {"black", "red", "blue", "green", "purple", "dark"},
        "purple": {"yellow", "green", "orange"},
        "orange": {"blue", "purple", "cyan"},
    },
    SIZE_CONTRADICTIONS={
        "small": {"large", "huge", "massive", "gigantic"},
        "large": {"small", "tiny", "little", "compact"},
    },
)

HIGH_RECALL_PROFILE = VerifierProfile(
    name="high_recall",
    version="1.0",
    description=(
        "Liberal verifier: high recall, lower precision. "
        "Use for coverage analysis or when false negatives are more costly than false positives."
    ),
    ATTRIBUTE_SIMILARITY_THRESHOLD=0.70,
    SPATIAL_TOLERANCE=0.20,
    OBJECT_PRESENCE_CONFIDENCE=0.75,
    COLOR_SYNONYMS={
        "red": {"crimson", "scarlet", "maroon", "pink", "rose", "salmon", "wine"},
        "blue": {"navy", "azure", "cobalt", "cyan", "teal", "aqua", "turquoise"},
        "green": {"lime", "olive", "sage", "forest", "mint", "jade", "emerald"},
        "yellow": {"golden", "amber", "gold", "lemon", "sunshine"},
        "black": {"charcoal", "ebony", "dark", "midnight"},
        "white": {"ivory", "cream", "off-white", "pearl", "snow"},
        "gray": {"grey", "silver", "ash", "slate", "stone"},
        "orange": {"coral", "burnt-orange", "peach", "apricot", "tangerine"},
        "purple": {"violet", "plum", "magenta", "lavender", "lilac"},
        "brown": {"tan", "beige", "khaki", "chocolate", "caramel", "copper"},
    },
    SIZE_SYNONYMS={
        "small": {"tiny", "little", "petite", "compact", "miniature"},
        "large": {"big", "huge", "massive", "gigantic", "enormous"},
        "medium": {"mid-sized", "moderate", "standard", "average"},
    },
    ACTION_SYNONYMS={
        "sit": {"sitting", "sits", "seated", "perching"},
        "stand": {"standing", "stands", "upright", "posing"},
        "walk": {"walking", "walks", "strolling", "pacing", "moving"},
        "run": {"running", "runs", "sprinting", "jogging", "rushing"},
        "lie": {"lying", "lays", "reclining", "sleeping", "resting"},
        "jump": {"jumping", "jumps", "leaping", "hopping"},
        "play": {"playing", "plays", "interacting"},
        "eat": {"eating", "eats", "munching"},
        "drink": {"drinking", "drinks", "sipping"},
    },
    COLOR_CONTRADICTIONS={
        "red": {"blue", "green", "yellow", "purple", "white", "black", "cyan", "navy"},
        "blue": {"red", "yellow", "orange", "brown", "black", "green", "gold"},
        "green": {"red", "pink", "purple", "orange", "blue"},
        "yellow": {"blue", "purple", "black", "navy"},
        "black": {"white", "yellow", "red", "purple", "bright"},
        "white": {"black", "red", "blue", "green", "purple", "dark"},
        "purple": {"yellow", "green", "orange", "lime"},
        "orange": {"blue", "purple", "cyan", "teal"},
    },
    SIZE_CONTRADICTIONS={
        "small": {"large", "huge", "massive", "gigantic", "enormous"},
        "large": {"small", "tiny", "little", "compact", "miniature"},
    },
)


# =============================================================================
# PROFILE REGISTRY & ACCESS
# =============================================================================

_PROFILE_REGISTRY: Dict[str, VerifierProfile] = {
    "strict": STRICT_PROFILE,
    "balanced": BALANCED_PROFILE,
    "high_recall": HIGH_RECALL_PROFILE,
}


def get_profile(name: str) -> VerifierProfile:
    """
    Retrieve a verifier profile by name.
    
    Args:
        name: Profile name ("strict", "balanced", "high_recall")
    
    Returns:
        VerifierProfile frozen dataclass
    
    Raises:
        KeyError: If profile name not found in registry
    """
    if name not in _PROFILE_REGISTRY:
        available = ", ".join(_PROFILE_REGISTRY.keys())
        raise KeyError(
            f"Profile '{name}' not found. Available profiles: {available}"
        )
    return _PROFILE_REGISTRY[name]


def list_profiles() -> Dict[str, str]:
    """List all available profiles with descriptions."""
    return {
        name: profile.description
        for name, profile in _PROFILE_REGISTRY.items()
    }


def register_profile(profile: VerifierProfile) -> None:
    """
    Register a custom verifier profile.
    
    Args:
        profile: VerifierProfile to register
    
    Raises:
        ValueError: If profile name already registered
    """
    if profile.name in _PROFILE_REGISTRY:
        raise ValueError(
            f"Profile '{profile.name}' already registered. "
            f"Use a different name or override explicitly."
        )
    _PROFILE_REGISTRY[profile.name] = profile
