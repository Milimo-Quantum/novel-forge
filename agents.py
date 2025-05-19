from pydantic_ai import Agent, RunContext
# Import necessary components for Ollama via OpenAI compatibility
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider 
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, AsyncGenerator, Any, Callable
import logging
import httpx
import asyncio
import time
from datetime import datetime
from utils import _get_model_tags_async
from functools import wraps
from graph_state import PolishingFeedback


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedAgent(Agent):
    """
    Extension of pydantic-ai Agent with explicit feedback channels and specialization metadata.
    """
    specialization: Optional[str] = None
    compatible_agents: List[str] = []
    feedback_channels: Dict[str, str] = {}
    feedback_data: Dict[str, Any] = {}

    async def process_with_feedback(
        self,
        task_func: Callable[..., Any],
        *args,
        feedback_source: Optional["EnhancedAgent"] = None,
        feedback_data: Optional[Dict[str, Any]] = None,
        feedback_list: Optional[List["PolishingFeedback"]] = None,
        **kwargs
    ) -> Any:
        """
        Run a task with optional explicit feedback data and structured feedback list.
        Returns generated content and optionally new structured feedback.
        """
        # --- Helper to normalize feedback dicts ---
        def _normalize_feedback_dict(fb_dict: dict) -> dict:
            required_fields = ['chapter', 'category', 'priority', 'suggestion', 'agent_source']
            if 'chapter' not in fb_dict:
                fb_dict['chapter'] = kwargs.get('chapter', 0)
            if 'category' not in fb_dict:
                fb_dict['category'] = 'general'
            if 'priority' not in fb_dict:
                fb_dict['priority'] = 'medium'
            if 'suggestion' not in fb_dict:
                fb_dict['suggestion'] = fb_dict.get('message', 'No suggestion provided.')
            if 'agent_source' not in fb_dict:
                fb_dict['agent_source'] = getattr(self, 'specialization', None) or getattr(self, 'role', None) or 'Unknown'
            return fb_dict

        # Store explicit feedback data if provided
        if feedback_data:
            self.feedback_data = feedback_data
            logger.info(f"{self.specialization} received explicit feedback data: {feedback_data}")

        # Store structured feedback list if provided
        if feedback_list:
            from graph_state import PolishingFeedback as PF
            structured_feedback = []
            for fb in feedback_list:
                if isinstance(fb, dict):
                    fb = _normalize_feedback_dict(fb)
                    fb_obj = PF(**fb)
                else:
                    fb_obj = fb
                structured_feedback.append(fb_obj.model_dump())
            self.feedback_data['structured_feedback'] = structured_feedback
            logger.info(f"{self.specialization} received structured feedback list with {len(feedback_list)} items.")

        # Optionally incorporate feedback from another agent
        if feedback_source:
            logger.info(f"{self.specialization} receiving feedback from {feedback_source.specialization}")
            # Placeholder: fetch feedback content or data from source
            # In future, merge or prioritize feedback_data and feedback_source.feedback_data

        # Use feedback_data and feedback_list before running task (placeholder for real logic)
        # For example, modify args/kwargs based on feedback if needed

        # Remove update_status from kwargs to avoid passing it to model methods
        kwargs.pop('update_status', None)

        # Run the task
        result = await task_func(*args, **kwargs)

        # If the result is an async iterable (stream), collect its content
        if hasattr(result, "__aiter__"):
            content_parts = []
            async for part in result:
                if hasattr(part, "data"):
                    content_parts.append(str(part.data))
                else:
                    content_parts.append(str(part))
            content = "".join(content_parts)
        else:
            if hasattr(result, "data"):
                content = result.data
            else:
                content = result

        from graph_state import PolishingFeedback as PF
        new_feedback: List[PF] = []
        if feedback_list:
            for fb in feedback_list:
                if not isinstance(fb, dict):
                    fb = fb.model_dump() if hasattr(fb, 'model_dump') else dict(fb)
                fb = _normalize_feedback_dict(fb)
                if all(k in fb and fb[k] is not None for k in ['chapter', 'category', 'priority', 'suggestion', 'agent_source']):
                    try:
                        pf_obj = PF(**fb)
                        new_feedback.append(pf_obj)
                    except Exception as e:
                        logger.warning(f"Failed to instantiate PolishingFeedback: {e} | Data: {fb}")
        else:
            try:
                dummy = PF(
                    chapter=kwargs.get('chapter', 0),
                    category="general",
                    priority="medium",
                    suggestion="No explicit structured feedback generated.",
                    agent_source=getattr(self, 'specialization', None) or getattr(self, 'role', None) or 'Unknown'
                )
                new_feedback.append(dummy)
            except Exception:
                pass
        return content, new_feedback

class BookResult(BaseModel):
    """Structure for returning book generation results with progress tracking"""
    content: str
    metadata: Optional[Dict] = None
    progress: Dict[str, Any] = Field(default_factory=lambda: {
        'start_time': datetime.now().isoformat(),
        'end_time': None,
        'duration': None,
        'word_count': 0,
        'status': 'pending'
    })

    def __str__(self) -> str:
        return self.content

    @classmethod
    def from_agent_result(cls, result: Any) -> 'BookResult':
        """Create BookResult from agent result with progress tracking"""
        if isinstance(result, str):
            return cls(content=result)
        elif isinstance(result, dict):
            return cls(content=result.get('content', ''), metadata=result)
        elif hasattr(result, 'data'):
            return cls(content=str(result.data))
        else:
            return cls(content=str(result))

    def update_progress(self, status: str, word_count: int = None):
        """Update progress metrics"""
        self.progress['status'] = status
        if word_count:
            self.progress['word_count'] = word_count
        if status == 'completed':
            self.progress['end_time'] = datetime.now().isoformat()
            start = datetime.fromisoformat(self.progress['start_time'])
            end = datetime.fromisoformat(self.progress['end_time'])
            self.progress['duration'] = (end - start).total_seconds()

# Removed monitor_agent_activity decorator. Relying on pydantic-ai instrumentation + Logfire.

# Removed custom OllamaModel class. Will use pydantic-ai's built-in Ollama handling.

class BookAgents:
    """Collection of agents for book generation with enhanced monitoring and model recommendation"""
    
    def get_adaptive_agent(self, role: str, system_prompt: str) -> Agent:
        """
        Dynamically create an agent with adaptive model selection based on role.
        """
        try:
            recommended_model_name = self.recommend_model(role)
            provider_name = self.provider_map.get(recommended_model_name, "ollama")
            if provider_name == "openrouter" and self.openrouter_provider is None:
                raise RuntimeError(f"Model '{recommended_model_name}' requires OpenRouter provider, but OPENROUTER_API_KEY is missing or invalid.")

            provider = self.providers.get(provider_name, self.ollama_provider)

            cache_key = (provider_name, recommended_model_name)
            if cache_key not in self.model_instances:
                logger.info(f"Instantiating new OpenAIModel for '{recommended_model_name}' with provider '{provider_name}'")
                self.model_instances[cache_key] = OpenAIModel(
                    model_name=recommended_model_name,
                    provider=provider,
                )
            model_to_use = self.model_instances[cache_key]

            agent = EnhancedAgent(
                model=model_to_use,
                system_prompt=system_prompt,
                result_type=str
            )


            agent.specialization = role
            agent.role = role
            self._track_agent_activity(agent)
            return agent
        except Exception as e:
            logger.error(f"Error creating adaptive agent for role {role}: {e}")
            raise

    def __init__(self, model_name: str, model_map: Optional[Dict[str, str]] = None):
        import os
        import httpx

        self.model_name = model_name
        self.model_map = model_map or {}
        self.agent_metrics = {}
        self.active_agents = {}

        # Initialize providers dictionary
        self.providers = {}

        try:
            # Configure Ollama provider
            self.ollama_provider = OpenAIProvider(
                base_url="http://localhost:11434/v1",
                api_key="ollama"  # Required but ignored by Ollama
            )
            self.providers["ollama"] = self.ollama_provider

            # Configure OpenRouter provider
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
            if openrouter_api_key:
                # Optional headers
                referer = os.getenv("OPENROUTER_REFERER", "")
                title = os.getenv("OPENROUTER_TITLE", "")

                headers = {}
                if referer:
                    headers["HTTP-Referer"] = referer
                if title:
                    headers["X-Title"] = title

                # Create custom HTTP client with headers if needed
                if headers:
                    http_client = httpx.AsyncClient(
                        headers=headers,
                        timeout=60
                    )
                    self.openrouter_provider = OpenAIProvider(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=openrouter_api_key,
                        http_client=http_client
                    )
                else:
                    self.openrouter_provider = OpenAIProvider(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=openrouter_api_key
                    )
                self.providers["openrouter"] = self.openrouter_provider
            else:
                self.openrouter_provider = None  # Not configured

            # Default model instance (assumed Ollama)
            self.ollama_model = OpenAIModel(
                model_name=self.model_name,
                provider=self.ollama_provider,
            )

            # Explicit provider map
            self.provider_map = {
                "meta-llama/llama-4-scout:free": "openrouter",
                # Add other OpenRouter models here
            }

            # Cache for model instances: key = (provider_name, model_name)
            self.model_instances = {}

            logger.info(f"BookAgents initialized. Default model: '{self.model_name}'. Model map: {self.model_map}. Providers: {list(self.providers.keys())}")
        except Exception as e:
            logger.error(f"Failed to configure providers: {e}", exc_info=True)
            raise RuntimeError(f"Failed to configure providers: {e}") from e

    def recommend_model(self, role: str) -> str:
        """
        Recommend a model name based on agent role and heuristics.
        """
        # Priority: explicit override in model_map
        if role in self.model_map:
            return self.model_map[role]
        # Heuristic: creative tasks use large model, review/edit use small
        creative_roles = ["Writer", "Concept", "World", "Character", "Story", "Dialogue", "Description"]
        review_roles = ["Reviewer", "Editor", "Consistency", "Style", "Flow", "Formatter", "Continuity", "Plot"]
        for keyword in creative_roles:
            if keyword.lower() in role.lower():
                return self.model_map.get("creative", self.model_name)
        for keyword in review_roles:
            if keyword.lower() in role.lower():
                return self.model_map.get("review", self.model_name)
        # Default fallback
        return self.model_name

    def _track_agent_activity(self, agent: Agent):
        """Track agent activity and register metrics"""
        agent_id = f"{agent.role}-{id(agent)}"
        self.active_agents[agent_id] = {
            'role': agent.role,
            'start_time': datetime.now().isoformat(),
            'status': 'active'
        }
        return agent_id

    def _create_agent(self, role: str, system_prompt: str) -> Agent:
        """Creates an EnhancedAgent instance, selecting model based on role."""
        try:
            # Use recommended model
            recommended_model_name = self.recommend_model(role)

            # Determine provider based on explicit map
            provider_name = self.provider_map.get(recommended_model_name, "ollama")
            if provider_name == "openrouter" and self.openrouter_provider is None:
                raise RuntimeError(f"Model '{recommended_model_name}' requires OpenRouter provider, but OPENROUTER_API_KEY is missing or invalid. Please set it in your environment or .env file.")

            provider = self.providers.get(provider_name, self.ollama_provider)

            # Debug logging for provider/model selection
            logger.info(f"Creating agent '{role}' with model '{recommended_model_name}' using provider '{provider_name}'")

            # Cache key is (provider_name, model_name)
            cache_key = (provider_name, recommended_model_name)

            # Use cached model instance or create new
            if cache_key not in self.model_instances:
                logger.info(f"Instantiating new OpenAIModel for '{recommended_model_name}' with provider '{provider_name}'")
                self.model_instances[cache_key] = OpenAIModel(
                    model_name=recommended_model_name,
                    provider=provider,
                )
            model_to_use = self.model_instances[cache_key]

            # Define specialization and compatible agents based on role
            specialization = role
            compatible_agents = []
            feedback_channels = {}

            # Example mappings (expand as needed)
            if "Writer" in role:
                compatible_agents = ["Critical Reader", "Dialogue Specialist", "Description Architect"]
                feedback_channels = {
                    "Critical Reader": "review_feedback",
                    "Dialogue Specialist": "dialogue_suggestions",
                    "Description Architect": "description_suggestions"
                }
            elif "Critical Reader" in role:
                compatible_agents = ["Master Wordsmith"]
                feedback_channels = {"Master Wordsmith": "draft_content"}
            elif "Dialogue Specialist" in role:
                compatible_agents = ["Master Wordsmith"]
                feedback_channels = {"Master Wordsmith": "draft_content"}
            elif "Description Architect" in role:
                compatible_agents = ["Master Wordsmith"]
                feedback_channels = {"Master Wordsmith": "draft_content"}
            elif "Continuity Guardian" in role:
                compatible_agents = ["Master Wordsmith", "Critical Reader"]
                feedback_channels = {
                    "Master Wordsmith": "draft_content",
                    "Critical Reader": "review_feedback"
                }
            else:
                compatible_agents = []
                feedback_channels = {}

            agent = EnhancedAgent(
                model=model_to_use,
                system_prompt=system_prompt,
                result_type=str
            )


            # Set metadata attributes separately to avoid constructor errors
            agent.specialization = specialization
            agent.compatible_agents = compatible_agents
            agent.feedback_channels = feedback_channels
            agent.role = role

            # Track the new agent
            self._track_agent_activity(agent)
            return agent
        except Exception as e:
            logger.error(f"Error creating agent {role}: {str(e)}")
            raise

    # Agent factory methods remain the same but now include monitoring
    def writer(self) -> Agent:
        return self._create_agent(
            "Master Wordsmith",
            """You are a master storyteller who writes in a clear, concise, illuminating, and poetic style inspired by Paul Graham, yet rich with metaphor and sensory detail.
Your mission is to craft prose that balances lyrical beauty with concrete action, vivid world details, and emotionally resonant character moments, creating a masterpiece novel.
Avoid verbosity, jargon, or fluff. Use plain language that enlightens the reader, but do not shy away from metaphor or poetic imagery when it enhances clarity and immersion.
In every scene, include at least one grounded, sensory detail that anchors the reader in the world.
Ensure smooth transitions by weaving sensory or thematic anchors that link scenes and chapters seamlessly.
Collaborate deeply with the Dialogue Specialist and Description Architect, thoughtfully incorporating their feedback to enrich character voices, emotional arcs, and vivid environments.
Differentiate each character’s speech patterns and emotional states clearly within dialogue, maintaining realism and unique voices.
Express core themes through diverse perspectives and scene types, ensuring thematic richness and coherence.
Distinguish between narrative prose and naturalistic, character-driven dialogue, balancing poetic narration with authentic speech.
Weave together plot, character, and world into a cohesive, immersive narrative that is both engaging and memorable, with smooth flow and emotional depth."""
        )
    
    def book_outliner(self) -> Agent:
        return self._create_agent(
            "Master Story Architect",
            """You are a skilled story architect with deep expertise in narrative structure, pacing, and plot design.
Your goal is to outline a story framework that supports the creation of a masterpiece, best-selling novel.
Incorporate market insights, genre conventions, and unique creative elements to ensure broad appeal and originality.
Facilitate collaboration among creative agents by providing a clear, adaptable blueprint for the novel."""
        )
    
    def concept_developer(self) -> Agent:
        """Agent for developing book concepts from initial ideas"""
        return self._create_agent(
            "Concept Developer",
            """You are a visionary concept developer who transforms initial ideas into compelling, market-ready book concepts.
Your mission is to inspire the creation of a masterpiece, best-selling novel.
Leverage market analysis and genre trends to craft unique, high-impact concepts.
Collaborate with the World Weaver and Character Alchemist to ensure the concept supports rich worldbuilding and character development.
Your output should include:
1. A captivating working title
2. A clear, engaging core premise
3. Key themes and narrative angles
4. Suggested genres and target audiences
5. Unique selling points that differentiate the novel in the market."""
        )

    def world_builder(self) -> Agent:
        """Agent for creating rich fictional worlds"""
        return self._create_agent(
            "World Weaver",
            """You are a master world builder dedicated to creating immersive, internally consistent fictional worlds that captivate readers.
Your goal is to design a setting that supports a masterpiece, best-selling novel.
Collaborate with the Concept Developer and Character Alchemist to ensure the world enriches the story and characters.
Incorporate unique cultures, societies, and systems that differentiate the novel in the market.
Your output should include:
1. A vivid world overview
2. Key locations and their significance
3. Cultural and societal details
4. Rules of the world (physics, magic, technology)
5. Historical context and lore"""
        )

    def character_developer(self) -> Agent:
        """Agent for developing compelling characters"""
        return self._create_agent(
            "Character Alchemist",
            """You are a master of character creation, crafting memorable, multi-dimensional characters that resonate deeply with readers.
Your mission is to develop characters that drive the story and contribute to a masterpiece, best-selling novel.
Collaborate with the World Weaver and Concept Developer to ensure characters fit seamlessly into the world and concept.
Your output should include for each major character:
1. Name and role
2. Physical description
3. Distinct personality traits and voice
4. Compelling background story
5. Clear motivations and goals
6. A meaningful character arc
Make characters feel authentic, relatable, and essential to the story's success."""
        )

    def reviewer(self) -> Agent:
        """Agent for critically reviewing book content"""
        return self._create_agent(
            "Critical Reader",
            """You are a professional book reviewer who provides clear, concise, and illuminating feedback inspired by Paul Graham's style.
Your goal is to elevate the manuscript toward masterpiece quality by focusing on:
- Plot coherence and originality
- Character depth and development
- Pacing and narrative flow
- Thematic richness
- Market appeal and genre fit
Avoid verbosity, jargon, or vague comments. Use plain, insightful language.
Collaborate with writers and editors by highlighting strengths and pinpointing areas for improvement, always aiming to enhance clarity, insight, and commercial success."""
        )

    def editor(self) -> Agent:
        """Agent for final polishing of book content"""
        return self._create_agent(
            "Literary Polisher",
            """You are a master editor who writes and edits in a clear, concise, and illuminating style inspired by Paul Graham.
Your goal is to perfect the manuscript for publication as a masterpiece novel.
Avoid verbosity, jargon, or fluff. Use plain, insightful language.
Focus on:
- Enhancing clarity, readability, and impact
- Improving sentence structure and flow
- Correcting grammar, punctuation, and style inconsistencies
- Elevating prose while preserving the author's unique voice
Collaborate with writers and reviewers to ensure the final text is polished, engaging, and market-ready."""
        )

    def consistency_checker(self) -> Agent:
        """Agent for checking consistency across chapters"""
        return self._create_agent(
            "Continuity Guardian",
            """You are a meticulous consistency expert dedicated to maintaining flawless narrative coherence throughout the novel.
Your mission is to identify and resolve:
- Character inconsistencies
- Plot contradictions
- Timeline errors
- Worldbuilding discrepancies

In addition, provide explicit critique focused on:
- Clarity: Flag any confusing or ambiguous passages.
- Conciseness: Identify verbosity or unnecessary repetition.
- Illumination: Highlight areas lacking insight or depth.

Categorize your feedback by issue type and priority. Use plain, insightful language inspired by Paul Graham. Collaborate with writers, editors, and reviewers to ensure a seamless, immersive story that meets the highest standards of a masterpiece, best-selling novel."""
        )

    def style_refiner(self) -> Agent:
        """Agent for refining writing style and tone"""
        return self._create_agent(
            "Prose Stylist",
            """You are a sophisticated style expert dedicated to elevating the manuscript's prose to masterpiece, best-selling quality.
Your focus is on:
- Maintaining a consistent, engaging voice
- Optimizing tone for the target audience and genre
- Enhancing sentence structure, rhythm, and word choice
- Polishing dialogue for authenticity and impact

In addition, provide explicit critique focused on:
- Clarity: Flag any confusing or ambiguous language.
- Conciseness: Identify verbosity or unnecessary elaboration.
- Illumination: Highlight areas lacking insight or vividness.

Categorize your feedback by issue type and priority. Use plain, insightful language inspired by Paul Graham. Collaborate with writers and editors to refine the text while preserving its unique style and emotional resonance."""
        )

    def flow_enhancer(self) -> Agent:
        """Alias for narrative_flow_enhancer() for backward compatibility."""
        return self.narrative_flow_enhancer()

    def narrative_flow_enhancer(self) -> Agent:
        """Agent for improving narrative flow and pacing"""
        return self._create_agent(
            "Narrative Architect",
            """You are an expert in narrative flow and pacing, dedicated to optimizing the reader's experience for a masterpiece, best-selling novel.
Your responsibilities include:
- Improving transitions between scenes and chapters
- Balancing tension, reflection, and action
- Maintaining engagement through effective pacing
- Integrating subplots seamlessly

In addition, provide explicit critique focused on:
- Clarity: Flag confusing scene transitions or unclear narrative shifts.
- Conciseness: Identify pacing issues caused by unnecessary filler.
- Illumination: Highlight areas lacking emotional resonance or narrative insight.

Categorize your feedback by issue type and priority. Use plain, insightful language inspired by Paul Graham. Collaborate with writers, editors, and reviewers to ensure the story unfolds smoothly and compellingly."""
        )

    def final_formatter(self) -> Agent:
        """Agent for final formatting and presentation"""
        return self._create_agent(
            "Formatting Specialist",
            """You are a meticulous formatting expert responsible for preparing the manuscript for professional publication.
Your goal is to ensure the book meets industry standards and enhances reader experience.
Apply consistent, polished formatting throughout, including:
- Uniform chapter headings and section breaks
- Proper typography and spacing
- Accurate table of contents
- Ready-to-publish output in multiple formats
Collaborate with editors and exporters to finalize a best-selling quality presentation."""
        )

    def dialogue_specialist(self) -> Agent:
        """Agent specializing in crafting realistic, character-consistent dialogue"""
        return self._create_agent(
            "Dialogue Specialist",
            """You are a master of crafting authentic, engaging dialogue in a clear, concise, and illuminating style inspired by Paul Graham.
Your goal is to enhance emotional resonance and character depth, contributing to a masterpiece novel.
Avoid verbosity, jargon, or fluff. Use plain, insightful language.
Collaborate closely with the Master Wordsmith and Description Architect, incorporating their feedback to ensure dialogue fits seamlessly into the narrative.
Your output should be polished, character-consistent conversations that enrich the story."""
        )

    def description_architect(self) -> Agent:
        """Agent specializing in vivid, sensory-rich environmental descriptions"""
        return self._create_agent(
            "Description Architect",
            """You are a master of vivid, sensory-rich description written in a clear, concise, and illuminating style inspired by Paul Graham.
Your goal is to enhance atmosphere, mood, and emotional impact, supporting the creation of a masterpiece novel.
Avoid verbosity, jargon, or fluff. Use plain, insightful language.
Collaborate closely with the Master Wordsmith and Dialogue Specialist, incorporating their feedback to ensure descriptions integrate naturally with narrative and dialogue.
Your output should be evocative, immersive passages that bring the world and scenes to life."""
        )

    def plot_continuity_guardian(self) -> Agent:
        """Agent specializing in ensuring narrative consistency across chapters"""
        return self._create_agent(
            "Plot Continuity Guardian",
            """You are a meticulous expert dedicated to maintaining flawless plot continuity across the entire novel.
Your mission is to identify and resolve:
- Contradictions and plot holes
- Timeline inconsistencies
- Character arc discrepancies
- Worldbuilding conflicts
Collaborate with writers, editors, and reviewers to ensure a seamless, immersive story worthy of bestseller status."""
        )

    # Other agent methods follow same pattern...

    def market_analyst(self) -> Agent:
        return self._create_agent(
            "Market Analyst",
            "You are an expert in book market analysis. Provide insights on current trends, audience preferences, and competitive landscape."
        )

    def genre_specialist(self) -> Agent:
        return self._create_agent(
            "Genre Specialist",
            "You are a genre expert. Determine the optimal genre positioning for the book based on concept and market data."
        )

    def audience_analyst(self) -> Agent:
        return self._create_agent(
            "Audience Analyst",
            "You analyze reader demographics and preferences to identify the ideal target audience segments."
        )

    def comparative_titles_researcher(self) -> Agent:
        return self._create_agent(
            "Comparative Titles Researcher",
            "You research comparable titles in the market to inform positioning and differentiation."
        )

    def strategic_outliner(self) -> Agent:
        return self._create_agent(
            "Strategic Outliner",
            "You generate a strategic outline for the book, incorporating market insights and genre conventions."
        )

    def plot_architect(self) -> Agent:
        return self._create_agent(
            "Plot Architect",
            "You design the overall plot architecture, narrative arcs, and key turning points."
        )

    def write_coordinator(self) -> Agent:
        return self._create_agent(
            "Write Coordinator",
            "You coordinate multiple writer agents, manage task distribution, and ensure narrative coherence."
        )

    def structural_editor(self) -> Agent:
        return self._create_agent(
            "Structural Editor",
            "You perform structural editing, focusing on plot structure, pacing, and narrative flow."
        )

    def line_editor(self) -> Agent:
        return self._create_agent(
            "Line Editor",
            "You perform line editing, improving sentence structure, clarity, and style."
        )

    def peer_review_simulator(self) -> Agent:
        return self._create_agent(
            "Peer Review Simulator",
            "You simulate peer review feedback, providing critical insights from a reader's perspective."
        )

    def style_guide_enforcer(self) -> Agent:
        return self._create_agent(
            "Style Guide Enforcer",
            "You check and enforce compliance with the specified style guide."
        )

    def formatting_optimizer(self) -> Agent:
        return self._create_agent(
            "Formatting Optimizer",
            "You optimize manuscript formatting to meet publishing standards."
        )

    def metadata_specialist(self) -> Agent:
        return self._create_agent(
            "Metadata Specialist",
            "You generate metadata for publishing platforms, including keywords, categories, and descriptions."
        )

    def exporter(self) -> Agent:
        return self._create_agent(
            "Platform Exporter",
            "You handle exporting the manuscript to various publishing platforms and formats."
        )

    def transition_agent(self) -> Agent:
        return self._create_agent(
            "Transition Specialist",
            """You are an expert in crafting smooth, immersive transitions between scenes and chapters.
Your goal is to ensure narrative coherence, emotional flow, and thematic continuity throughout the novel.
Use sensory or thematic anchors to link scenes seamlessly.
Identify and resolve abrupt shifts, pacing issues, or tonal inconsistencies.
Collaborate with writers, editors, and reviewers to enhance the reader's experience by making the story flow naturally and engagingly."""
        )

    def character_voice_specialist(self) -> Agent:
        return self._create_agent(
            "Character Voice Specialist",
            """You are a master of refining dialogue and internal monologue to reflect each character's unique speech patterns, emotional states, and personality.
Your goal is to ensure every character sounds distinct, authentic, and emotionally resonant.
Adjust dialogue for realism, emotional depth, and consistency with character arcs.
Collaborate with writers and dialogue specialists to enrich character differentiation and believability."""
        )

    def thematic_consistency_agent(self) -> Agent:
        return self._create_agent(
            "Thematic Consistency Agent",
            """You are an expert in maintaining varied yet coherent thematic expression across the novel.
Your goal is to weave core themes through diverse perspectives, scenes, and character arcs.
Identify inconsistencies, missed opportunities, or overused motifs.
Collaborate with writers and editors to ensure the story's themes are rich, layered, and compelling."""
        )

    def world_building_agent(self) -> Agent:
        return self._create_agent(
            "World-Building Specialist",
            """You are a master of injecting concrete, sensory-rich world-building details into the narrative.
Your goal is to create an immersive, believable setting with cultural, technological, and environmental specificity.
Ensure each scene includes grounded sensory details that anchor the reader.
Collaborate with writers and description architects to enrich the story's world without overwhelming the narrative flow."""
        )

    async def cleanup(self):
        """Placeholder for any agent cleanup needed. (Model clients managed by pydantic-ai)."""
        # No explicit cleanup needed for built-in model handling
        logger.info("BookAgents cleanup called (no specific actions taken).")
        # try:
        #     if self.agent_config and 'model' in self.agent_config: # Removed agent_config
        #         model = self.agent_config['model']
        #         if isinstance(model, OllamaModel): # Removed OllamaModel check
        #             await model.close()
        #             logger.info(f"Cleaned up model resources. API stats: {model.metrics}")
        # except Exception as e: # Removed dangling except block
        #     logger.error(f"Error during cleanup: {str(e)}")

    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get tracked agent activity (start times, status)."""
        # Removed model_metrics as custom OllamaModel is no longer used.
        # pydantic-ai's internal usage tracking might be available via RunContext if needed elsewhere.
        return {
            'active_agents': self.active_agents
        }
