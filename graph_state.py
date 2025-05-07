from pydantic import BaseModel, Field, EmailStr
from typing import List, Dict, Optional, Any, Callable, Literal, Set, Union
from datetime import datetime
import time
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Forward reference for agent messages if needed, though likely imported
# from pydantic_ai.messages import ModelMessage

# --- New Models for Enhanced Polishing ---

class PolishingFeedback(BaseModel):
    """Structured feedback from polishing agents."""
    chapter: int
    category: str # 'consistency' | 'style' | 'flow' | 'formatting'
    priority: str # 'critical' | 'high' | 'medium' | 'low'
    suggestion: str
    agent_source: str # e.g., 'Consistency Checker', 'Style Refiner'
    before: Optional[str] = None # Optional context before change
    after: Optional[str] = None  # Optional context after change

class ChapterContext(BaseModel):
    """Contextual information extracted from or relevant to a chapter."""
    key_characters: List[str] = Field(default_factory=list)
    plot_developments: List[str] = Field(default_factory=list)
    worldbuilding_elements: List[str] = Field(default_factory=list)
    emotional_tone: Optional[str] = None
    pace: Optional[str] = None # e.g., 'fast', 'medium', 'slow'

class StyleGuide(BaseModel):
    """Overall style guidelines for the book."""
    target_reading_level: Optional[str] = None # e.g., 'Young Adult', 'Adult'
    narrative_perspective: Optional[str] = None # e.g., 'First Person', 'Third Person Limited'
    dialogue_style: Optional[str] = None # e.g., 'Formal', 'Colloquial'
    descriptive_richness: Optional[str] = None # e.g., 'Sparse', 'Moderate', 'Rich'
    overall_tone: Optional[str] = None # e.g., 'Serious', 'Humorous', 'Suspenseful'

class AppliedChange(BaseModel):
    """Record of a change applied based on feedback."""
    feedback_id: Optional[str] = None # Link back to PolishingFeedback if possible
    chapter: int
    description: str
    timestamp: float = Field(default_factory=time.time)

# --- Publishing Metadata Model ---
class PublishingMetadata(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    copyright: Optional[str] = None
    dedication: Optional[str] = None
    about_author: Optional[str] = None
    isbn: Optional[str] = None
    genre: Optional[str] = None
    keywords: Optional[str] = None
    description: Optional[str] = None
    publisher: Optional[str] = None
    plot_outline: Optional[str] = None
    pacing_labels: Optional[List[str]] = None

    # Additional dynamic metadata fields
    scene_transitions: Optional[Any] = None
    market_analysis: Optional[Any] = None
    genre_positioning: Optional[Any] = None
    audience: Optional[Any] = None
    comparable_titles: Optional[Any] = None
    strategic_outline: Optional[Any] = None
    plot_architecture: Optional[Any] = None
    proposal: Optional[Any] = None
    metadata: Optional[Any] = None
    final_report: Optional[Any] = None
    epub_path: Optional[str] = None
    epub_export_status: Optional[str] = None
    pdf_path: Optional[str] = None
    pdf_export_status: Optional[str] = None
    docx_path: Optional[str] = None
    docx_export_status: Optional[str] = None

# --- Existing Models ---
class CharacterProfile(BaseModel):
    """Represents a single character profile."""
    name: str
    role: str
    description: str
    motivations: str
    background: str = ""  # Character's history and backstory

class WritingStats(BaseModel):
    """Tracks writing progress metrics."""
    total_words_written: int = 0
    total_writing_time: float = 0.0
    average_wpm: int = 0
    completed_chapters: int = 0
    chapters_approved: int = 0
    arc: Optional[str] = None

class ChapterResult(BaseModel):
    """Represents the content and metadata of a single VERSION of a chapter."""
    chapter_number: int
    version: int = 1 # Track chapter version
    title: Optional[str] = None
    content: str
    word_count: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: Literal["draft", "review_pending", "needs_rewrite", "approved"] = "draft" # More specific status
    file_path: Optional[str] = None # Path for this specific version's markdown file
    revision_notes: Optional[str] = None # Notes specific to this revision
    feedback: List[PolishingFeedback] = Field(default_factory=list) # Feedback received for this version

    # --- Enhancements for multi-perspective and references ---
    pov_character: Optional[str] = None  # Point of view character
    pov_provenance: Optional[str] = "unknown"  # 'explicit', 'heuristic', or 'unknown'
    color_code: Optional[str] = None     # Color tag for visualization
    tags: List[str] = Field(default_factory=list)  # Arbitrary tags (e.g., themes, arcs)
    references: Dict[str, str] = Field(default_factory=dict)  # Cross-references (e.g., {"world_element_id": "desc"})

    def __hash__(self):
        # Hash based on chapter number and version for uniqueness
        return hash((self.chapter_number, self.version))

    def __eq__(self, other):
        # Ensure equality checks both chapter number and version
        if not isinstance(other, ChapterResult):
            return NotImplemented
        return self.chapter_number == other.chapter_number and self.version == other.version

class AgentMetrics(BaseModel):
    """Metrics for a single agent's activity."""
    total_time: float = 0.0
    call_count: int = 0
    last_call_timestamp: Optional[float] = None
    tokens_used: int = 0 # Placeholder, needs integration with model usage

class SystemResourceSnapshot(BaseModel):
    """Snapshot of system resources at a point in time."""
    timestamp: float
    cpu_percent: Optional[float] = None
    memory_usage: Optional[float] = None # e.g., percentage

class GenerationConfig(BaseModel):
    """Configuration settings for the book generation."""
    model_name: str
    num_chapters: int
    min_words_per_chapter: int
    temperature: float = 0.7
    # --- User-configurable iteration and feedback limits ---
    max_iterations: int = 2  # Default: 2 iterations per agent/node
    max_feedback_items: int = 6  # Default: 6 feedback/context items per agent/node

class BookGenerationState(BaseModel):
    node: Optional[Any] = None
    """Central state object for the book generation graph."""
    run_id: str = Field(default_factory=lambda: f"run_{int(time.time())}")
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    current_status: str = "Initializing" # e.g., Initializing, Generating Title, Writing Chapter 3, Reviewing, Complete, Error
    error_message: Optional[float] = None

    # --- Enriched Input Templates ---
    three_act_structure: Optional[Dict[str, str]] = Field(default_factory=dict)  # keys: 'act1', 'act2', 'act3'
    character_bios: Dict[str, Dict[str, str]] = Field(default_factory=dict)  # character_name -> {'desire': str, 'fear': str, 'relationship': str}
    world_rules: List[str] = Field(default_factory=list)  # e.g., ['Magic drains stamina', 'AI requires rituals']

    # --- Flags and Issues ---
    flags: Dict[str, List[str]] = Field(default_factory=dict)  # e.g., {'chapter_1': ['Missing sensory anchor', ...]}

    # --- Style Prompt Cache ---
    style_prompt: Optional[str] = None  # Stores Paul Graham style instructions to prepend to prompts

    # --- Version Control ---
    version_history: Dict[str, Dict[str, Any]] = Field(default_factory=dict, exclude=True)
    
    def save_version(self, label: Optional[str] = None, note: Optional[str] = None):
        """Save current combined draft as a version snapshot."""
        timestamp = datetime.now().isoformat()
        label = label or timestamp
        content = self.get_full_draft()
        # Save a deep copy of the entire state for rollback
        import copy
        snapshot = copy.deepcopy(self.model_dump())
        self.version_history[label] = {
            "content": content,
            "note": note,
            "timestamp": timestamp,
            "state_snapshot": snapshot
        }
        logger.info(f"Saved version '{label}' with note: {note}")

    def get_version(self, label: str) -> Optional[str]:
        """Retrieve content of a saved version."""
        version = self.version_history.get(label)
        return version.get("content") if version else None

    def list_versions(self) -> List[str]:
        """List all saved version labels."""
        return list(self.version_history.keys())

    def diff_versions(self, label1: str, label2: str) -> str:
        """Return a simple diff between two versions."""
        import difflib
        v1 = self.get_version(label1) or ""
        v2 = self.get_version(label2) or ""
        diff = difflib.unified_diff(
            v1.splitlines(), v2.splitlines(),
            fromfile=label1, tofile=label2, lineterm=""
        )
        return "\n".join(diff)

    def annotate_version(self, label: str, note: str):
        """Add or update annotation for a saved version."""
        if label in self.version_history:
            self.version_history[label]["note"] = note
            logger.info(f"Annotated version '{label}' with note: {note}")

    def restore_version(self, label: str):
        """Restore the entire state from a saved checkpoint."""
        version = self.version_history.get(label)
        if not version or "state_snapshot" not in version:
            logger.warning(f"No snapshot found for version '{label}'")
            return
        snapshot = version["state_snapshot"]
        # Restore all fields from snapshot
        for key, value in snapshot.items():
            try:
                setattr(self, key, value)
            except Exception:
                pass
        logger.info(f"Restored state from checkpoint '{label}'")

    def restore_chapter_version(self, chapter_num: int, version_num: int):
        """Restore a specific chapter to a previous version."""
        versions = self.chapter_versions.get(chapter_num, [])
        target_version = None
        for v in versions:
            if v.version == version_num:
                target_version = v
                break
        if not target_version:
            logger.warning(f"Chapter {chapter_num} version {version_num} not found for restore.")
            return
        # Remove all later versions
        self.chapter_versions[chapter_num] = [
            v for v in versions if v.version <= version_num
        ]
        # Update current status
        self.current_chapter_status[chapter_num] = target_version.status
        logger.info(f"Restored Chapter {chapter_num} to version {version_num}")

    # --- Input & Config ---
    initial_concept: str
    config: GenerationConfig

    # --- Summaries for context management ---
    chapter_summaries: Dict[int, str] = Field(default_factory=dict)
    combined_summary: Optional[str] = None

    # --- Generated Artifacts ---
    book_title: Optional[str] = None
    refined_concept: Optional[str] = None
    extracted_elements: Optional[str] = None # Added field for extracted elements
    book_outline: Optional[str] = None # Could be structured (e.g., Dict[int, str])
    world_details: Optional[str] = None # Could be structured
    characters: List[CharacterProfile] = Field(default_factory=list)

    # --- Publishing Metadata & Front/Back Matter ---
    publishing_metadata: PublishingMetadata = Field(default_factory=PublishingMetadata)
    copyright_page: Optional[str] = None
    dedication_page: Optional[str] = None
    about_author_page: Optional[str] = None
    acknowledgments_page: Optional[str] = None
    front_matter: Optional[str] = None
    back_matter: Optional[str] = None
    export_format: Optional[str] = None  # e.g., 'epub', 'pdf', 'docx'
    export_status: Optional[str] = None  # e.g., 'pending', 'in_progress', 'completed', 'failed'
    export_path: Optional[str] = None

    # --- Chapter & Version Management ---
    chapter_versions: Dict[int, List[ChapterResult]] = Field(default_factory=dict) # Stores all versions of each chapter, keyed by chapter number
    current_chapter_status: Dict[int, Literal["draft", "review_pending", "needs_rewrite", "approved"]] = Field(default_factory=dict) # Tracks latest status per chapter
    review_feedback: Optional[str] = None # Could be structured
    final_book_content: Optional[str] = None
    final_book_path: Optional[str] = None
    has_been_saved: bool = False  # Track if book has been saved
    final_stats_path: Optional[str] = None
    review_cycle_count: int = 0  # Track number of review cycles
    max_review_cycles: int = 3  # Maximum allowed review cycles

    # --- Process Tracking ---
    # chapters_to_rewrite is now handled by the 'needs_rewrite' status in current_chapter_status
    current_chapter_writing: Optional[int] = None # Track which chapter is actively being written/rewritten

    # --- Metrics & Monitoring ---
    total_words_generated: int = 0
    writing_stats: WritingStats # Removed default_factory, initialized explicitly in book_crew.py
    agent_metrics: Dict[str, AgentMetrics] = Field(default_factory=dict) # Keyed by agent role/name
    system_resources: List[SystemResourceSnapshot] = Field(default_factory=list)

    # --- Agent Message History (Optional, could be large) ---
    # Consider if storing full history here is feasible or if it should be managed elsewhere/persisted separately
    # agent_message_history: Dict[str, List[ModelMessage]] = Field(default_factory=dict) # Keyed by agent role/name

    # --- Enhanced Polishing & Combined Draft State ---
    chapter_contexts: Dict[int, ChapterContext] = Field(default_factory=dict)
    style_guidelines: Optional[StyleGuide] = None # Can be populated during concept/outline phase
    feedback_history: List[PolishingFeedback] = Field(default_factory=list) # Overall feedback from polishing agents
    applied_changes: Dict[int, List[AppliedChange]] = Field(default_factory=dict) # Keyed by chapter number
    combined_draft_path: Optional[str] = None # Path to the latest combined draft file
    combined_draft_version: int = 0 # Version of the combined draft

    # --- User Interaction Enhancements ---
    freeze_flags: Dict[str, bool] = Field(default_factory=lambda: {
        "concept": False,
        "outline": False,
        "characters": False,
        "world": False,
        "chapters": False
    })
    user_feedback: Dict[str, str] = Field(default_factory=dict)  # Arbitrary user feedback or instructions

    # --- UI Callback (Not persisted) ---
    progress_callback: Optional[Callable] = Field(default=None, exclude=True)

    def update_progress(self, status: str, message: Optional[str] = None):
        """Helper to update the overall status."""
        self.current_status = status
        if message:
            logger.info(f"Progress update: {status} - {message}")

    def record_agent_activity(self, agent_name: str, duration: float, tokens: int = 0):
        """Update metrics for a specific agent."""
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = AgentMetrics()
        metrics = self.agent_metrics[agent_name]
        metrics.call_count += 1
        metrics.total_time += duration
        metrics.last_call_timestamp = time.time()
        metrics.tokens_used += tokens # Add token tracking

    def record_system_resources(self, cpu: Optional[float], memory: Optional[float]):
        """Add a snapshot of system resources."""
        self.system_resources.append(
            SystemResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu,
                memory_usage=memory
            )
        )

    def add_chapter_version(self, chapter_result: ChapterResult):
        """Adds a new version of a chapter result."""
        chapter_num = chapter_result.chapter_number
        if chapter_num not in self.chapter_versions:
            self.chapter_versions[chapter_num] = []

        # Remove older version if it exists with the same version number (e.g., during resume/rewrite)
        self.chapter_versions[chapter_num] = [
            ch for ch in self.chapter_versions[chapter_num]
            if ch.version != chapter_result.version
        ]
        self.chapter_versions[chapter_num].append(chapter_result)
        # Sort versions for consistency
        self.chapter_versions[chapter_num].sort(key=lambda ch: ch.version)

        # Update current status for this chapter
        self.current_chapter_status[chapter_num] = chapter_result.status

        # Recalculate total words based on the latest version of each chapter
        self.total_words_generated = sum(
            self.get_latest_chapter_version(num).word_count
            for num in self.chapter_versions
            if self.get_latest_chapter_version(num) # Ensure chapter exists
        )
        self.update_progress(f"Chapter {chapter_num} v{chapter_result.version} Saved ({chapter_result.status})")

    def get_chapter_version(self, chapter_num: int, version: int) -> Optional[ChapterResult]:
        """Gets a specific version of a chapter."""
        if chapter_num in self.chapter_versions:
            for ch in self.chapter_versions[chapter_num]:
                if ch.version == version:
                    return ch
        return None

    def get_latest_chapter_version(self, chapter_num: int) -> Optional[ChapterResult]:
        """Gets the most recent version of a chapter."""
        if chapter_num in self.chapter_versions and self.chapter_versions[chapter_num]:
            # Versions are sorted when added, so the last one is the latest
            return self.chapter_versions[chapter_num][-1]
        return None

    def validate_chapter_content(self, content: str) -> bool:
        """Validate chapter content matches book concept."""
        if not self.refined_concept:
            return True  # No concept to validate against
            
        concept_keywords = set(self.refined_concept.lower().split())
        content_keywords = set(content.lower().split())
        return len(concept_keywords & content_keywords) >= 3  # At least 3 matching keywords

    def get_latest_approved_chapter(self, chapter_num: int) -> Optional[ChapterResult]:
        """Gets the latest approved version of a chapter, or falls back to latest draft."""
        if chapter_num not in self.chapter_versions or not self.chapter_versions[chapter_num]:
            return None

        # Prefer latest approved version matching concept
        for version in reversed(self.chapter_versions[chapter_num]):
            if version.status == "approved" and self.validate_chapter_content(version.content):
                return version

        # Fallback: latest version regardless of status
        latest = self.get_latest_chapter_version(chapter_num)
        if latest:
            logger.warning(f"No approved version found for chapter {chapter_num}, falling back to latest draft v{latest.version} with status '{latest.status}'")
            return latest

        logger.warning(f"No versions found at all for chapter {chapter_num}")
        return None

    def get_full_draft(self) -> str:
        """Alias for get_combined_draft_content() for backward compatibility."""
        return self.get_combined_draft_content()

    def get_combined_draft_chapters(self) -> List[ChapterResult]:
        """Get all latest chapter versions for the combined draft."""
        return [
            self.get_latest_chapter_version(i)
            for i in range(1, self.config.num_chapters + 1)
            if self.get_latest_chapter_version(i)
        ]

    def get_combined_draft_content(self, version_to_combine: Literal["latest", "approved"] = "latest") -> str:
        """Combines content of chapters based on specified version status."""
        draft_parts = []
        for i in range(1, self.config.num_chapters + 1):
            chapter_to_add = None
            latest_version = self.get_latest_chapter_version(i)

            if latest_version:
                if version_to_combine == "latest":
                    chapter_to_add = latest_version
                elif version_to_combine == "approved" and latest_version.status == "approved":
                     chapter_to_add = latest_version

            if chapter_to_add:
                # Add a simple chapter heading
                draft_parts.append(f"# Chapter {i}\n\n{chapter_to_add.content}")
            else:
                # Indicate missing or non-qualifying chapters
                status = self.current_chapter_status.get(i, "missing")
                if version_to_combine == "approved" and status != "approved":
                    status = f"{status} (not approved)"
                elif not latest_version:
                     status = "missing"

                draft_parts.append(f"\n\n--- CHAPTER {i} ({status.upper()}) NOT INCLUDED ---\n\n")

        return "\n\n".join(draft_parts)

    @classmethod
    def from_json_with_concept_check(cls, data: dict, expected_concept: str):
        """
        Load state from JSON, enforcing that the initial concept matches the expected concept.
        Raises ValueError if there is a mismatch.
        """
        state = cls.model_validate(data)
        # Defensive: also check for refined concept drift
        from utils import logger
        if getattr(state, 'initial_concept', None) != expected_concept:
            logger.error(f"[ConceptFlow] State deserialization mismatch! Loaded initial_concept={getattr(state, 'initial_concept', None)!r}, expected={expected_concept!r}")
            raise ValueError("Loaded state concept does not match expected run concept")
        # No concept integrity check: always allow refined concept as canonical
        return state

    class Config:
        # If using Pydantic v2, this helps with potential complex types
        arbitrary_types_allowed = True
