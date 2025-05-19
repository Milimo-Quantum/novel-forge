import os
import json
import logging
import time
import asyncio
import re
from typing import Optional, Callable, Any, Dict
from pathlib import Path
import datetime
from pydantic_graph import Graph, GraphRunContext  
from pydantic_graph.persistence.file import FileStatePersistence  

from agents import BookAgents # BookAgents will be our dependency
from utils import load_style_prompt  # Import style prompt loader
# Import necessary models from graph_state
from graph_state import BookGenerationState, GenerationConfig, AgentMetrics, ChapterResult, WritingStats # Added WritingStats
from graph_nodes import (
    StartGeneration,
    PlotScaffolding,
    PacingControl,
    SceneTransitionGenerator,
    MarketAnalysis,
    GenrePositioning,
    AudienceTargeting,
    ComparativeTitles,
    StrategicOutlineGenerator,
    GenerateProposal,
    GenerateTitle,
    DevelopConcept,
    PlotArchitect,
    CreateOutline,
    BuildWorld,
    DevelopCharacters,
    RefineWorldWithCharacters,
    RefineCharactersWithWorld,
    WriteCoordinator,
    WriteChapter,
    VerifyCrossReferences,
    GenerateSummaries,
    StructuralEditor,
    LineEditor,
    PeerReviewSimulator,
    MultiStageReviewHub,
    EvaluateQualityMetrics,
    StyleGuideEnforcer,
    FormattingOptimizer,
    MarketMetadataGenerator,
    GenerateFrontMatter,
    GenerateBackMatter,
    AssembleBook,
    PolishBook,
    FormatBook,
    SaveFinalBook,
    ReviewBook,
    PeerReview,
    EditorialReview,
    ConsistencyCheck,
    StyleRefinement,
    FlowEnhancement,
    ReviewAggregator,
    ExportEPUB,
    ExportPDF,
    ExportDOCX,
    FinalReport,
    End,  # Import End for type hints
)
from utils import log_operation # Keep if still relevant for the run method

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the graph structure (flattened, no nested lists)
book_generation_graph = Graph[BookGenerationState, BookAgents, str](
    nodes=[
        StartGeneration,
        DevelopConcept,
        PlotScaffolding,
        PacingControl,
        SceneTransitionGenerator,
        MarketAnalysis,
        GenrePositioning,
        AudienceTargeting,
        ComparativeTitles,
        StrategicOutlineGenerator,
        GenerateProposal,
        GenerateTitle,
        PlotArchitect,
        CreateOutline,
        BuildWorld,
        DevelopCharacters,
        RefineWorldWithCharacters,
        RefineCharactersWithWorld,
        WriteCoordinator,
        WriteChapter,
        ReviewBook,
        ReviewAggregator,
        VerifyCrossReferences,
        GenerateSummaries,
        StructuralEditor,
        LineEditor,
        PeerReviewSimulator,
        MultiStageReviewHub,
        EvaluateQualityMetrics,
        StyleGuideEnforcer,
        FormattingOptimizer,
        MarketMetadataGenerator,
        GenerateFrontMatter,
        GenerateBackMatter,
        PeerReview,
        EditorialReview,
        ConsistencyCheck,
        StyleRefinement,
        FlowEnhancement,
        AssembleBook,
        PolishBook,
        FormatBook,
        SaveFinalBook,
        ExportEPUB,
        ExportPDF,
        ExportDOCX,
        FinalReport,
    ]
)

class BookCrew:

    async def resume_from_checkpoint(self, checkpoint_label: str) -> str:
        """
        Resume the graph execution from a saved checkpoint label.
        """
        try:
            # Load all saved states
            history = await self.persistence.load_all()
            # Find the snapshot with the given label
            target_snapshot = None
            for snapshot in history:
                if snapshot.label == checkpoint_label:
                    target_snapshot = snapshot
                    break
            if not target_snapshot:
                logger.error(f"Checkpoint '{checkpoint_label}' not found.")
                return f"Checkpoint '{checkpoint_label}' not found."

            # Restore the state from snapshot
            state = target_snapshot.state
            logger.info(f"Restored state from checkpoint '{checkpoint_label}'.")

            # Save restored state as current
            await self.persistence.save(state)

            # Resume graph execution from this state
            async with book_generation_graph.iter_from_persistence(
                persistence=self.persistence, deps=self.agents
            ) as run:
                run.state.progress_callback = self.progress_callback
                while True:
                    next_node = await run.next()
                    if next_node is None:
                        break
                    if isinstance(next_node, End):
                        logger.info(f"Graph completed with End node: {next_node}")
                        break
                    if isinstance(next_node, list):
                        await asyncio.gather(*[
                            node.run(GraphRunContext(state=run.state, deps=run.deps)) for node in next_node
                        ])
                    else:
                        await next_node.run(GraphRunContext(state=run.state, deps=run.deps))
                    logger.info(f"Graph node executed: {type(next_node).__name__}")
                    await self._stream_progress(run.state)
                run_result = run.result

            logger.info(f"Resumed graph execution from checkpoint '{checkpoint_label}' completed.")
            return "Resume completed."

        except Exception as e:
            logger.error(f"Failed to resume from checkpoint '{checkpoint_label}': {e}", exc_info=True)
            return f"Failed to resume from checkpoint '{checkpoint_label}': {e}"
    # Class-level configuration (can be moved to GenerationConfig defaults)
    BASE_URL = 'http://localhost:11434'
    DEFAULT_MODEL = "meta-llama/llama-4-scout:free"
    MIN_CHAPTERS = 1
    MAX_CHAPTERS = 50
    DEFAULT_CHAPTERS = 3 # Reduced default for quicker testing
    DEFAULT_MIN_WORDS = 500 # Reduced default

    def adaptive_model_selection(self, task_type: str, content_length: int = 0, complexity: int = 1) -> str:
        """
        Selects the best model dynamically based on task type, content length, and complexity.
        """
        # Example heuristic:
        # - For very long or complex tasks, prefer larger models
        # - For quick reviews or short content, prefer smaller models
        # - Use benchmarking data if available (future enhancement)

        # Default to current map if exists
        current_model = self.model_map.get(task_type)
        if current_model:
            return current_model

        # Heuristic selection
        if task_type in ["writer", "concept_developer", "book_outliner", "world_builder", "character_developer"]:
            if content_length > 1000 or complexity > 2:
                return self.config.model_name
            else:
                return self.config.model_name
        elif task_type in ["reviewer", "editor", "consistency_checker", "style_refiner", "flow_enhancer", "final_formatter"]:
            if content_length > 2000:
                return self.config.model_name
            else:
                return self.config.model_name
        else:
            return self.DEFAULT_MODEL

    def update_model_map(self, task_type: str, content_length: int = 0, complexity: int = 1):
        """
        Update the model_map dynamically for a given task.
        """
        selected_model = self.adaptive_model_selection(task_type, content_length, complexity)
        self.model_map[task_type] = selected_model
        logger.info(f"Adaptive model selected for {task_type}: {selected_model}")

    def __init__(
        self,
        initial_concept: str, # Changed from input_file to concept string
        num_chapters: Optional[int] = None,
        model_name: str = DEFAULT_MODEL,
        progress_callback: Optional[Callable] = None,
        min_words_per_chapter: int = DEFAULT_MIN_WORDS,
        temperature: float = 0.7,
        run_id: Optional[str] = None, # Allow specifying a run ID for resuming
        model_map: Optional[Dict[str, str]] = None, # New: task-specific model overrides
        checkpoint_frequency: int = 1 # New: save checkpoint every N steps
    ):
        """Initialize the BookCrew to manage graph-based book generation."""
        self.initial_concept = initial_concept
        self.progress_callback = progress_callback
        self.run_id = run_id or f"run_{int(time.time())}"

        # If no model_map provided, create one with intelligent defaults
        if not model_map:
            model_map = {}
            # Assign large model for creative tasks
            creative_model = model_name
            # Assign smaller/faster model for review tasks
            fast_model = model_name
            model_map.update({
                "default": creative_model,
                "writer": creative_model,
                "concept_developer": creative_model,
                "book_outliner": creative_model,
                "world_builder": creative_model,
                "character_developer": creative_model,
                "reviewer": fast_model,
                "editor": fast_model,
                "consistency_checker": fast_model,
                "style_refiner": fast_model,
                "flow_enhancer": fast_model,
                "final_formatter": fast_model,
                "dialogue_specialist": fast_model,
                "description_architect": fast_model,
                "plot_continuity_guardian": fast_model
            })
        self.model_map = model_map or {}
        self.checkpoint_frequency = max(1, checkpoint_frequency)

        # Validate and store configuration
        validated_num_chapters = max(
            min(num_chapters or self.DEFAULT_CHAPTERS, self.MAX_CHAPTERS),
            self.MIN_CHAPTERS
        )
        self.config = GenerationConfig(
            model_name=model_name,
            num_chapters=validated_num_chapters,
            min_words_per_chapter=min_words_per_chapter,
            temperature=temperature,
        )

        # Initialize Agents (Dependencies for the graph)
        # Error handling during agent init is crucial
        try:
            # Use task-specific model if provided, else default
            selected_model = self.model_map.get("default", self.config.model_name)
            self.agents = BookAgents(model_name=selected_model, model_map=self.model_map)
        except Exception as e:
            logger.error(f"Failed to initialize BookAgents: {e}", exc_info=True)
            # Propagate error or handle appropriately
            raise RuntimeError(f"Agent initialization failed: {e}") from e

        # Setup directories (can be done within graph nodes if preferred)
        self.draft_dir = os.path.join('novelForge', 'books', 'drafts')
        self.persistence_dir = os.path.join('novelForge', 'runs')
        os.makedirs(self.draft_dir, exist_ok=True)
        os.makedirs(self.persistence_dir, exist_ok=True)

        # Setup Persistence
        self.persistence_path = Path(self.persistence_dir) / f"{self.run_id}.json"
        self.persistence = FileStatePersistence(self.persistence_path)
        # Set graph types for persistence to handle serialization/deserialization
        self.persistence.set_graph_types(book_generation_graph)

        # Variables to store final results from the graph run
        self.final_book_path: Optional[str] = None
        self.final_stats_path: Optional[str] = None
        self.error_message: Optional[str] = None

    async def _stream_progress(self, state: BookGenerationState):
        """Extracts progress info from state and calls the callback."""
        # Save checkpoint at key stages
        try:
            # Save checkpoint every N steps or at key stages
            save_due_to_stage = state.current_status in [
                "World Built",
                "Characters Developed",
                "World Refined with Characters",
                "Characters Refined with World",
                "Title Generated",
                "Concept Developed",
                "Outline Created",
                "Reviewing Book",
                "Review Complete - All Approved",
                "Polishing Complete",
                "Formatting Complete",
                "Book Saved"
            ] or (state.current_status.startswith("Writing Chapter") and "Complete" not in state.current_status)

            # Initialize step counter if not present
            if not hasattr(self, "_step_counter"):
                self._step_counter = 0
            self._step_counter += 1

            if save_due_to_stage or (self._step_counter % self.checkpoint_frequency == 0):
                # Compose a descriptive label
                label = f"{state.current_status} @ {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                note = f"Auto checkpoint at {state.current_status}"
                state.save_version(label=label, note=note)
                logger.info(f"Checkpoint saved: {label}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint at {state.current_status}: {e}")

        if not self.progress_callback:
            return

        # Granular weighted progress calculation
        weights = {
            "title": 5,
            "concept": 10,
            "outline": 10,
            "world": 10,
            "characters": 10,
            "chapters": 40,
            "review": 15
        }

        # New: assign incremental progress based on current_status string
        percent = 0
        status = state.current_status or ""
        if "Developing Concept" in status:
            percent = 5
        elif "Generating Title" in status:
            percent = 10
        elif "Creating Outline" in status:
            percent = 15
        elif "Building World" in status:
            percent = 20
        elif "Developing Characters" in status:
            percent = 25

        # Add weighted progress based on completed fields
        if state.book_title:
            percent += weights["title"]
        if state.refined_concept:
            percent += weights["concept"]
        if state.book_outline:
            percent += weights["outline"]
        if state.world_details:
            percent += weights["world"]
        if state.characters:
            percent += weights["characters"]

        completed_chapters = len([
            num for num in state.chapter_versions
            if state.current_chapter_status.get(num) in ('draft', 'approved')
        ])
        
        total_chapters = state.config.num_chapters or 1
        chapter_progress = (completed_chapters / total_chapters) * weights["chapters"]
        percent += chapter_progress

        if state.review_feedback:
            percent += weights["review"]

        percent = min(100, int(percent))

        if state.current_status == "Complete":
            percent = 100
        elif state.current_status == "Error":
            percent = state.agent_metrics.get("System", AgentMetrics()).call_count  # Placeholder or last known %

        # Extract relevant stats for the callback
        # Align this with what the Streamlit app expects
        # Build stats dictionary with proper formatting
        current_content = ""
        if state.current_chapter_writing:
            latest_chapter = state.get_latest_chapter_version(state.current_chapter_writing)
            if latest_chapter:
                current_content = latest_chapter.content

        chapter_metrics = []
        for num in sorted(state.chapter_versions):
            chapter = state.get_latest_chapter_version(num)
            if chapter:
                # Fix: assign 'Unknown' if pov_character missing
                if not getattr(chapter, 'pov_character', None):
                    chapter.pov_character = "Unknown"
                chapter_metrics.append(chapter.model_dump())

        last_chapter_stats = None
        if state.chapter_versions:
            last_chapter = state.get_latest_chapter_version(max(state.chapter_versions.keys()))
            if last_chapter:
                last_chapter_stats = last_chapter.model_dump()

        # Calculate elapsed time
        elapsed_seconds = time.time() - state.start_time
        elapsed_minutes = elapsed_seconds / 60 if elapsed_seconds > 0 else 0

        # Calculate words per minute
        words_per_minute = 0
        if elapsed_minutes > 0:
            words_per_minute = state.total_words_generated / elapsed_minutes

        # Calculate average WPM from writing_stats if available
        average_wpm = getattr(state.writing_stats, 'average_wpm', 0)

        # Total writing time
        total_writing_time = getattr(state.writing_stats, 'total_writing_time', elapsed_seconds)

        # Build stats dictionary
        stats_for_callback = {
            'current_content': current_content,
            'words_generated': state.total_words_generated,
            'words_per_minute': words_per_minute,
            'completed_chapters': completed_chapters,
            'total_chapters': state.config.num_chapters,
            'elapsed_time': elapsed_seconds,
            'agent_metrics': {
                name: metrics.model_dump()
                for name, metrics in state.agent_metrics.items()
            },
            'chapter_metrics': chapter_metrics,
            'system_resources': (
                state.system_resources[-1].model_dump()
                if state.system_resources
                else None
            ),
            'last_chapter_stats': last_chapter_stats,
            'streaming': False,
            'total_writing_time': total_writing_time,
            'average_wpm': average_wpm,
            # --- emit final paths if available ---
            'final_book_path': getattr(state, 'final_book_path', None),
            'final_stats_path': getattr(state, 'final_stats_path', None),
            # --- emit export paths ---
            'epub_path': getattr(state.publishing_metadata, 'epub_path', None),
            'pdf_path': getattr(state.publishing_metadata, 'pdf_path', None),
            'docx_path': getattr(state.publishing_metadata, 'docx_path', None),
            # --- emit export statuses ---
            'epub_export_status': getattr(state.publishing_metadata, 'epub_export_status', None),
            'pdf_export_status': getattr(state.publishing_metadata, 'pdf_export_status', None),
            'docx_export_status': getattr(state.publishing_metadata, 'docx_export_status', None),
        }


        try:
            # Note: The original callback expected agent_name, which isn't directly available here.
            # We pass the overall status as the 'stage'.
            await self.progress_callback(
                stage=state.current_status,
                message=f"Step: {state.current_status}", # Simple message
                agent_name="GraphRunner", # Generic name for graph progression
                percent=percent,
                stats=stats_for_callback
            )
        except Exception as e:
            logger.error(f"Error in progress callback: {e}", exc_info=True)

    # @log_operation # Keep decorator if desired
    async def run(self) -> str:
        """Runs the book generation graph, handling state and progress."""
        logger.info(f"Starting graph run: {self.run_id} with config: {self.config}")
        run_result = None
        final_output = "Graph execution did not complete."

        try:
            # Check if persistence file exists to resume
            if self.persistence_path.exists():
                logger.info(f"Resuming graph run from persistence file: {self.persistence_path}")
                # Use iter_from_persistence to resume
                async with book_generation_graph.iter_from_persistence(
                    persistence=self.persistence, deps=self.agents
                ) as run:
                    run.state.progress_callback = self.progress_callback # Assign callback to state
                    while True:
                        next_node = await run.next()
                        if next_node is None:
                            break
                        if isinstance(next_node, End):
                            logger.info(f"Graph completed with End node: {next_node}")
                            break
                        if isinstance(next_node, list):
                            # Parallel branch execution
                            await asyncio.gather(*[
                                node.run(GraphRunContext(state=run.state, deps=run.deps)) for node in next_node
                            ])
                        else:
                            await next_node.run(GraphRunContext(state=run.state, deps=run.deps))
                        logger.info(f"Graph node executed: {type(next_node).__name__}")
                        # Stream progress immediately after each node execution
                        await self._stream_progress(run.state)
                    run_result = run.result # Get the final result after iteration completes
            else:
                logger.info(f"Starting new graph run: {self.run_id}")
                # Initialize state and start node for a new run
                start_node = StartGeneration(
                    initial_concept=self.initial_concept,
                    config=self.config
                )
                
                # Check for existing state to preserve has_been_saved flag
                initial_state = BookGenerationState(
                    run_id=self.run_id,
                    initial_concept=self.initial_concept,
                    config=self.config,
                    writing_stats=WritingStats(background=self.initial_concept)
                )
                
                if self.persistence_path.exists():
                    try:
                        history = await self.persistence.load_all()
                        if history:
                            prev_state = history[-1].state
                            initial_state.has_been_saved = prev_state.has_been_saved
                    except Exception as e:
                        logger.warning(f"Failed to load previous state: {e}")
                # Load Paul Graham style prompt and assign to state
                initial_state.style_prompt = load_style_prompt()

                await book_generation_graph.initialize(
                    node=start_node,
                    state=initial_state,
                    # deps=self.agents, # Removed: deps are passed during run/iter
                    persistence=self.persistence
                )
                logger.info("Graph initialized and first state persisted.")

                # Emit initial progress immediately after initialization
                await self._stream_progress(initial_state)

                # Now iterate starting from the persisted state
                async with book_generation_graph.iter_from_persistence(
                    persistence=self.persistence, deps=self.agents
                ) as run:
                    run.state.progress_callback = self.progress_callback # Assign callback to state
                    while True:
                        next_node = await run.next()
                        if next_node is None:
                            break
                        if isinstance(next_node, End):
                            logger.info(f"Graph completed with End node: {next_node}")
                            break
                        if isinstance(next_node, list):
                            # Parallel branch execution
                            await asyncio.gather(*[
                                node.run(GraphRunContext(state=run.state, deps=run.deps)) for node in next_node
                            ])
                        else:
                            await next_node.run(GraphRunContext(state=run.state, deps=run.deps))
                        logger.info(f"Graph node executed: {type(next_node).__name__}")
                        # Stream progress immediately after each node execution
                        await self._stream_progress(run.state)
                    run_result = run.result

                # Process final result
                if run_result:
                    # Get final state from the last snapshot in history FIRST
                    history = await self.persistence.load_all()
                    final_state = history[-1].state if history else None

                    # Check for error message in the final state to determine success
                    if final_state and final_state.error_message:
                        # --- Handle Failure ---
                        error_msg = f"Graph run {self.run_id} failed. Reason: {final_state.error_message}"
                        self.error_message = final_state.error_message
                        logger.error(error_msg)
                        final_output = error_msg
                        # Send error progress update using the final state
                        await self._stream_progress(final_state)
                    elif final_state:
                        # --- Handle Success ---
                        final_output = run_result.output # Should be the final book path
                        self.final_book_path = final_output
                        self.final_stats_path = getattr(final_state, 'final_stats_path', None) # Use getattr for safety
                        logger.info(f"Graph run {self.run_id} completed successfully. Output: {final_output}")
                        # Final progress update using the final state
                        await self._stream_progress(final_state)
                    else:
                        # --- Handle case where final state couldn't be loaded ---
                        error_msg = f"Graph run {self.run_id} finished, but final state could not be loaded from persistence."
                        self.error_message = error_msg
                        logger.error(error_msg)
                        final_output = error_msg
                else:
                    logger.error(f"Graph run {self.run_id} did not produce a final result.")
                    final_output = "Graph run finished without a result."
                    self.error_message = final_output


        except Exception as e:
            logger.error(f"Error during graph execution {self.run_id}: {e}", exc_info=True)
            self.error_message = f"Critical error during graph execution: {e}"
            final_output = self.error_message
            # Try to update progress with error state if possible
            try:
                history = await self.persistence.load_all()
                final_state = history[-1].state if history else None
                if final_state:
                    final_state.current_status = "Error" # Update status in the loaded state
                    final_state.error_message = self.error_message
                    await self._stream_progress(final_state)
            except Exception as progress_err:
                logger.error(f"Failed to send final error progress update: {progress_err}")

        finally:
            # Clean up agent resources
            logger.info("Cleaning up agent resources...")
            await self.agents.cleanup()

        return final_output

    # Removed old methods: _read_book_concept, _sanitize_filename, _get_system_resources,
    # update_progress, _process_progress_updates, generate_title, _run_with_error_handling
    # Their logic is now encapsulated within graph nodes and state management.
