from typing import Dict, List, Optional
from pydantic import BaseModel
import json

# Import necessary models from graph_state
from graph_state import CharacterProfile, StyleGuide, BookGenerationState

class ChapterContent(BaseModel):
    """Structure for chapter content"""
    title: str
    content: str
    revision_notes: Optional[str] = None

class TaskPrompts:
    """Collection of prompt templates for book generation tasks"""
    
    @staticmethod
    def _assert_no_placeholder(prompt: str):
        # List any forbidden placeholder names/phrases
        forbidden = ["Elara Vance"]
        for word in forbidden:
            assert word not in prompt, f"Prompt contains forbidden placeholder: {word}"
        return prompt

    @staticmethod
    def market_analysis(concept: str) -> dict:
        prompt = f"""Analyze the current book market for the following concept:

{concept}

Provide insights on:
- Market trends
- Audience preferences
- Competitive landscape
- Opportunities and challenges

Return a detailed market analysis."""
        return {"prompt": TaskPrompts._assert_no_placeholder(prompt)}

    @staticmethod
    def genre_positioning(concept: str) -> dict:
        prompt = f"""Determine the optimal genre positioning for this book concept:

{concept}

Consider:
- Genre conventions
- Market fit
- Audience expectations

Return a recommended genre positioning."""
        return {"prompt": TaskPrompts._assert_no_placeholder(prompt)}

    @staticmethod
    def audience_targeting(concept: str) -> dict:
        prompt = f"""Identify the ideal target audience segments for this book concept:

{concept}

Include:
- Demographics
- Interests
- Reading preferences

Return a detailed audience profile."""
        return {"prompt": TaskPrompts._assert_no_placeholder(prompt)}

    @staticmethod
    def comparative_titles(concept: str) -> dict:
        prompt = f"""Research comparable titles for this book concept:

{concept}

List:
- Title
- Author
- Genre
- Key similarities/differences

Return a list of comparable titles with brief notes."""
        return {"prompt": TaskPrompts._assert_no_placeholder(prompt)}

    @staticmethod
    def strategic_outline(concept: str, market_data: str) -> dict:
        prompt = f"""Generate a strategic outline for this book concept, incorporating market insights:

CONCEPT:
{concept}

MARKET DATA:
{market_data}

Outline key elements, themes, and structure aligned with market opportunities."""
        return {"prompt": TaskPrompts._assert_no_placeholder(prompt)}

    @staticmethod
    def plot_architecture(concept: str) -> dict:
        prompt = f"""Design the overall plot architecture for this book concept:

{concept}

Include:
- Main narrative arcs
- Key turning points
- Climax and resolution
- Subplots

Return a detailed plot architecture."""
        return {"prompt": TaskPrompts._assert_no_placeholder(prompt)}

    @staticmethod
    def write_coordination() -> dict:
        prompt = """Coordinate multiple writer agents, assign tasks, and ensure narrative coherence.

Return a coordination plan."""
        return {"prompt": prompt}

    @staticmethod
    def structural_editing(content: str) -> dict:
        prompt = f"""Perform structural editing on the following manuscript:

{content}

Focus on:
- Plot structure
- Pacing
- Scene order
- Narrative flow

Return a report with suggested structural changes."""
        return {"prompt": prompt}

    @staticmethod
    def line_editing(content: str) -> dict:
        prompt = f"""Perform line editing on the following manuscript:

{content}

Focus on:
- Sentence structure
- Clarity
- Style
- Grammar

Return the edited text or a report with suggestions."""
        return {"prompt": prompt}

    @staticmethod
    def peer_review_simulation(content: str) -> dict:
        prompt = f"""Simulate peer review feedback for this manuscript:

{content}

Provide:
- Overall impressions
- Strengths
- Weaknesses
- Suggestions

Return detailed peer review feedback."""
        return {"prompt": prompt}

    @staticmethod
    def style_guide_enforcement(content: str, style_guide: str) -> dict:
        prompt = f"""Check the following manuscript for compliance with this style guide:

STYLE GUIDE:
{style_guide}

MANUSCRIPT:
{content}

Return a report of style guide violations and suggestions."""
        return {"prompt": prompt}

    @staticmethod
    def formatting_optimization(content: str) -> dict:
        prompt = f"""Optimize formatting of this manuscript for publishing standards:

{content}

Apply:
- Consistent headings
- Proper spacing
- Clean typography

Return the formatted manuscript."""
        return {"prompt": prompt}

    @staticmethod
    def metadata_generation(concept: str) -> dict:
        prompt = f"""Generate publishing metadata for this book concept:

{concept}

Include:
- Keywords
- Categories
- Description
- Author info

Return metadata in JSON format."""
        return {"prompt": TaskPrompts._assert_no_placeholder(prompt)}

    @staticmethod
    def export_manuscript(content: str) -> dict:
        prompt = f"""Prepare this manuscript for export to publishing platforms:

{content}

Generate:
- Platform-specific formats
- Export instructions

Return export-ready files or instructions."""
        return {"prompt": prompt}

    @staticmethod
    def extract_elements(concept: str) -> dict:
        prompt = f"""Extract key elements from this book concept:
        {concept}

        Identify and return:
        1. Character names and roles
        2. Key themes
        3. World/setting details
        4. Plot points
        5. Style/tone indicators

        Format as JSON with these keys:
        - characters
        - themes
        - settings
        - plot_points
        - style_indicators"""
        return {"prompt": TaskPrompts._assert_no_placeholder(prompt)}

    @staticmethod
    def develop_concept(concept: str, elements: dict = None, publishing_metadata: Optional[dict] = None) -> dict:
        meta_str = publishing_metadata.model_dump_json(indent=2) if publishing_metadata else "None provided"
        prompt = f"""Analyze and enhance this book concept while preserving all key elements:

        ORIGINAL CONCEPT:
        {concept}

        EXTRACTED ELEMENTS:
        {elements if elements else 'None provided'}

        MARKET & STRATEGIC INSIGHTS:
        {meta_str}

        Provide a refined and expanded version that:
        1. Preserves all extracted elements exactly
        2. Incorporates relevant market and strategic insights
        3. Only fills in gaps where needed
        4. Maintains original style/tone indicators
        5. Expands on existing elements rather than replacing them
        6. Clearly marks any new additions

        Return the enhanced concept in markdown format with sections for:
        - Preserved Elements (unchanged)
        - Enhanced Elements (expanded)
        - New Additions (clearly marked)"""
        return {"prompt": TaskPrompts._assert_no_placeholder(prompt)}

    @staticmethod
    def generate_title(concept: str) -> dict:
        prompt = f"""Generate a single, catchy book title based on this concept:
        CONCEPT:
        {concept}

        Return ONLY the title text, with no extra formatting, labels, or explanation.
        Maximum 10 words.
        """
        return {"prompt": TaskPrompts._assert_no_placeholder(prompt)}

    @staticmethod
    def create_outline(concept: str, num_chapters: int, publishing_metadata: Optional[dict] = None) -> dict:
        meta_str = publishing_metadata.model_dump_json(indent=2) if publishing_metadata else "None provided"
        prompt = f"""Create a detailed chapter-by-chapter outline for a book based on the following concept:

        CONCEPT:
        {concept}

        MARKET & STRATEGIC INSIGHTS:
        {meta_str}

        Generate an outline for exactly {num_chapters} chapters.
        For each chapter, include:
        1. Chapter title
        2. Main plot points
        3. Character development
        4. Scene descriptions
        5. Thematic elements

        Structure this as a clear chapter-by-chapter breakdown in markdown format."""
        return {"prompt": TaskPrompts._assert_no_placeholder(prompt)}

    @staticmethod
    def build_world(outline: str, concept: str, publishing_metadata: Optional[dict] = None) -> dict:
        meta_str = publishing_metadata.model_dump_json(indent=2) if publishing_metadata else "None provided"
        prompt = f"""Create a detailed world-building document based on:

        OUTLINE:
        {outline}

        CONCEPT:
        {concept}

        MARKET & STRATEGIC INSIGHTS:
        {meta_str}

        Include:
        1. Physical environment (geography, climate, etc.)
        2. Cultural elements
        3. Historical background
        4. Social structures
        5. Rules/systems (magic, technology, etc.)
        6. Important locations

        Make this world rich and internally consistent in markdown format."""
        return {"prompt": prompt}

    @staticmethod
    def develop_characters(outline: str, world: str, publishing_metadata: Optional[dict] = None) -> dict:
        meta_str = publishing_metadata.model_dump_json(indent=2) if publishing_metadata else "None provided"
        prompt = f"""Create detailed character profiles based on:

        OUTLINE:
        {outline}

        WORLD:
        {world}

        MARKET & STRATEGIC INSIGHTS:
        {meta_str}

        For each major character, provide:
        1. Physical description
        2. Personality traits
        3. Background/history
        4. Motivations/goals
        5. Character arc
        6. Relationships with other characters
        7. Role in the story

        Format the character profiles in markdown."""
        return {"prompt": prompt}

    @staticmethod
    def write_chapter(
        outline: str,
        characters: str,
        world: str,
        chapter_num: int,
        previous_chapter_content: str,
        min_words: int,
        state: BookGenerationState
    ) -> dict:
        # Context management: truncate long contexts
        def truncate_text(text: str, max_words: int = 1000) -> str:
            words = text.split()
            if len(words) > max_words:
                return " ".join(words[:max_words]) + " ...[truncated]"
            return text

        outline = truncate_text(outline)
        characters = truncate_text(characters)
        world = truncate_text(world)
        prev_summary = truncate_text(state.combined_summary or "No summary available.")

        # Extract enriched inputs
        act_structure = state.three_act_structure or {}
        bios = state.character_bios or {}
        world_rules = state.world_rules or []

        # Compose bios summary
        bios_summary = "\n".join(
            f"{name}: Desire={bio.get('desire','')}, Fear={bio.get('fear','')}, Relationship={bio.get('relationship','')}"
            for name, bio in bios.items()
        ) or "None provided."

        # Compose world rules summary
        world_rules_text = "\n".join(f"- {rule}" for rule in world_rules) or "None provided."

        prompt = f"""You are writing Chapter {chapter_num} of the novel '{state.book_title or "Untitled"}'.

CORE CONCEPT:
{state.refined_concept or state.initial_concept}

THREE ACT STRUCTURE:
Act 1 Goal: {act_structure.get('act1','Not specified')}
Act 2 Conflict: {act_structure.get('act2','Not specified')}
Act 3 Resolution: {act_structure.get('act3','Not specified')}

CHARACTER BIOS:
{bios_summary}

WORLD RULES:
{world_rules_text}

CONTEXT:
Overall Outline: {outline}
Characters: {characters}
World Details: {world}
Previous Chapters Summary: {prev_summary}

ADDITIONAL ELEMENTS:
- Character Profiles: {[c.name for c in state.characters] if state.characters else "None"}
- World Details: {state.world_details or "None"}
- Style Guidelines: {state.style_guidelines or "None"}

TASK: Write the full content for Chapter {chapter_num}.
Ensure the chapter is at least {min_words} words long.
Maintain strict consistency with all concept elements.

**MANDATORY CONSTRAINTS:**

- **Narrative Cohesion:**  
  Use sensory anchors (sound, smell, object) to link scenes.  
  Follow the act structure, focusing on the chapter's specific goal/conflict/resolution.  
  Include a clear chapter-level goal.

- **Character Development:**  
  Reflect each character's desire, fear, and relationships in their actions and dialogue.  
  Differentiate speech patterns (e.g., terse antagonist, emotional ally, wise mentor).  
  Reveal motivations through behavior, not exposition.

- **World-Building:**  
  Integrate at least one tactile/sensory detail per scene.  
  Respect world rules (e.g., magic drains stamina).  
  Show cultural-tech/magic fusion where relevant.

- **Thematic Depth:**  
  Illustrate themes via character choices and consequences.  
  Tie stakes to personal relationships, not just global events.

- **Pacing:**  
  Structure as: Goal → Obstacle → Choice → Consequence.  
  Alternate introspective and active scenes.

- **Emotional Resonance:**  
  Use symbolic objects to evoke emotion.  
  Include at least one vulnerability moment (e.g., trembling hands).

Write this chapter in a polished, professional style using markdown formatting.  
Explicitly mark the POV character at the end as:  
`POV: Character Name`

**ADDITIONAL CONSTRAINTS:**

- **Motif and Phrase Variation:**  
  Avoid overusing the same metaphors, motifs, or phrases (e.g., "fracture", "song", "grief", "eclipse"). Use varied language and fresh imagery.

- **Emotional Tone Dynamics:**  
  Vary emotional intensity across scenes. Include moments of quiet, humor, or reflection to balance intense passages.

- **Scene Objectives and Conflict:**  
  For each scene, clearly define the character's immediate goal, the obstacle, the conflict, and the consequence.

- **Character Voice Differentiation:**  
  Ensure each character's dialogue and internal thoughts reflect their unique background, motivations, and emotional state.

- **Balance Style and Clarity:**  
  While maintaining poetic elements, prioritize clarity, plot advancement, and grounded sensory detail.
"""
        return {"prompt": prompt}

    @staticmethod
    def write_chapter_prompt(state: BookGenerationState, chapter_num: int, min_words: int) -> dict:
        """
        Compose a prompt for book chapter writing, always using the latest valid concept and all relevant context.
        """
        outline = truncate_text(state.book_outline)
        characters = truncate_text(str(state.characters))
        world = truncate_text(state.world_details)
        prev_summary = truncate_text(state.combined_summary or "No summary available.")
        act_structure = state.three_act_structure or {}
        bios = state.character_bios or {}
        world_rules = state.world_rules or []
        bios_summary = "\n".join(
            f"{name}: Desire={bio.get('desire','')}, Fear={bio.get('fear','')}, Relationship={bio.get('relationship','')}"
            for name, bio in bios.items()
        ) or "None provided."
        world_rules_text = "\n".join(f"- {rule}" for rule in world_rules) or "None provided."
        # Always use the latest valid concept
        concept = state.refined_concept or state.initial_concept
        prompt = f"""You are writing Chapter {chapter_num} of the novel '{state.book_title or "Untitled"}'.\n\nCORE CONCEPT:\n{concept}\n\nTHREE ACT STRUCTURE:\nAct 1 Goal: {act_structure.get('act1','Not specified')}\nAct 2 Conflict: {act_structure.get('act2','Not specified')}\nAct 3 Resolution: {act_structure.get('act3','Not specified')}\n\nCHARACTER BIOS:\n{bios_summary}\n\nWORLD RULES:\n{world_rules_text}\n\nCONTEXT:\nOverall Outline: {outline}\nCharacters: {characters}\nWorld Details: {world}\nPrevious Chapters Summary: {prev_summary}\n\nADDITIONAL ELEMENTS:\n- Character Profiles: {[c.name for c in state.characters] if state.characters else "None"}\n- World Details: {state.world_details or "None"}\n- Style Guidelines: {state.style_guidelines or "None"}\n\nTASK: Write the full content for Chapter {chapter_num}.\nEnsure the chapter is at least {min_words} words long.\nMaintain strict consistency with all concept elements.\n\n**MANDATORY CONSTRAINTS:**\n\n- **Narrative Cohesion:**  \n  Use sensory anchors (sound, smell, object) to link scenes.  \n  Follow the act structure, focusing on the chapter's specific goal/conflict/resolution.  \n  Include a clear chapter-level goal.\n\n- **Character Development:**  \n  Reflect each character's desire, fear, and relationships in their actions and dialogue.  \n  Differentiate speech patterns (e.g., terse antagonist, emotional ally, wise mentor).  \n  Reveal motivations through behavior, not exposition.\n\n- **World-Building:**  \n  Integrate at least one tactile/sensory detail per scene.  \n  Respect world rules (e.g., magic drains stamina).  \n  Show cultural-tech/magic fusion where relevant.\n\n- **Thematic Depth:**  \n  Illustrate themes via character choices and consequences.  \n  Tie stakes to personal relationships, not just global events.\n\n- **Pacing:**  \n  Structure as: Goal → Obstacle → Choice → Consequence.  \n  Alternate introspective and active scenes.\n\n- **Emotional Resonance:**  \n  Use symbolic objects to evoke emotion.  \n  Include at least one vulnerability moment (e.g., trembling hands).\n\nWrite this chapter in a polished, professional style using markdown formatting.  \nExplicitly mark the POV character at the end as:  \n`POV: Character Name`\n\n**ADDITIONAL CONSTRAINTS:**\n\n- **Motif and Phrase Variation:**  \n  Avoid overusing the same metaphors, motifs, or phrases (e.g., "fracture", "song", "grief", "eclipse"). Use varied language and fresh imagery.\n\n- **Emotional Tone Dynamics:**  \n  Vary emotional intensity across scenes. Include moments of quiet, humor, or reflection to balance intense passages.\n\n- **Scene Objectives and Conflict:**  \n  For each scene, clearly define the character's immediate goal, the obstacle, the conflict, and the consequence.\n\n- **Character Voice Differentiation:**  \n  Ensure each character's dialogue and internal thoughts reflect their unique background, motivations, and emotional state.\n\n- **Balance Style and Clarity:**  \n  While maintaining poetic elements, prioritize clarity, plot advancement, and grounded sensory detail.\n"""
        return {"prompt": TaskPrompts._assert_no_placeholder(prompt)}

    @staticmethod
    def review_book(content: str) -> dict:
        prompt = f"""Review this manuscript:

        {content}

        Provide a comprehensive critique focusing on:
        1. Plot coherence and pacing
        2. Character development and consistency
        3. World-building integration
        4. Thematic depth
        5. Writing style and tone
        6. Potential improvements

        Be specific and constructive in your feedback. Format your review in markdown."""
        return {"prompt": prompt}

    @staticmethod
    def polish_book(content: str, review: str) -> dict:
        prompt = f"""Polish this manuscript based on the review:

        MANUSCRIPT:
        {content}

        REVIEW NOTES:
        {review}

        Focus on:
        1. Addressing reviewer feedback
        2. Enhancing prose quality
        3. Improving flow and pacing
        4. Strengthening character moments
        5. Tightening plot elements
        6. Polishing dialogue
        7. Ensuring consistency

        Maintain the original voice while elevating the overall quality. Use markdown formatting for the output."""
        return {"prompt": prompt}

    @staticmethod
    def check_consistency(draft: str, characters: str, world: str, num_chapters: int) -> dict:
        prompt = f"""Analyze the following book draft ({num_chapters} chapters) for internal consistency.

        DRAFT:
        {draft}

        CHARACTERS:
        {characters}

        WORLD DETAILS:
        {world}

        Focus on:
        1. Character Consistency: Traits, appearances, relationships, motivations.
        2. Plot Consistency: Timeline contradictions, plot holes, event mismatches.
        3. World Consistency: Rules (magic, physics, society), locations, descriptions.
        4. Continuity Errors: Specific detail contradictions.

        **Output Format:**
        Provide your feedback as a markdown list. Each item should represent a distinct piece of feedback.
        For each feedback item, specify:
        - **Chapter:** The chapter number(s) the feedback primarily applies to (e.g., "Chapter 5", "Chapters 3-7", "General"). Use "General" if it applies broadly.
        - **Category:** "Consistency"
        - **Priority:** Critical, High, Medium, Low (Estimate the severity).
        - **Suggestion:** A clear description of the inconsistency or error found.

        Example:
        ```markdown
        - **Chapter:** Chapter 5
          **Category:** Consistency
          **Priority:** High
          **Suggestion:** Character X's eye color was described as blue in Chapter 2 but brown here.

        - **Chapter:** General
          **Category:** Consistency
          **Priority:** Medium
          **Suggestion:** The travel time between City A and City B seems inconsistent across different chapters.
        ```

        If no inconsistencies are found, return the single line: "No major inconsistencies found."
        """
        return {"prompt": prompt}

    @staticmethod
    def refine_style(draft: str, style_guide: Optional[str], num_chapters: int) -> dict:
        style_context = f"STYLE GUIDELINES:\n{style_guide}\n\n" if style_guide else "No specific style guidelines provided.\n\n"
        prompt = f"""Analyze and suggest improvements for the writing style of the following draft ({num_chapters} chapters).

        DRAFT:
        {draft}

        {style_context}
        Focus on:
        1. Consistent Tone & Voice: Suitability for genre/audience.
        2. Sentence Structure: Awkward phrasing, run-ons, lack of variety.
        3. Word Choice: Weak verbs, clichés, repetition, jargon.
        4. Clarity & Readability: Confusing or unclear sections.
        5. Dialogue Polish: Naturalness, character voice.

        **Output Format:**
        Provide your feedback as a markdown list. Each item should represent a distinct piece of feedback.
        For each feedback item, specify:
        - **Chapter:** The chapter number(s) the feedback primarily applies to (e.g., "Chapter 8", "Chapters 10-12", "General"). Use "General" if it applies broadly.
        - **Category:** "Style"
        - **Priority:** High, Medium, Low (Estimate impact on quality).
        - **Suggestion:** A clear description of the style issue and a suggestion for improvement (provide specific examples from the text where possible).

        Example:
        ```markdown
        - **Chapter:** Chapter 8
          **Category:** Style
          **Priority:** Medium
          **Suggestion:** The description of the marketplace uses repetitive adjectives like 'busy' and 'crowded'. Consider more varied sensory details.

        - **Chapter:** General
          **Category:** Style
          **Priority:** High
          **Suggestion:** Overuse of passive voice throughout the manuscript weakens the narrative drive. Example in Chapter 3: "The door was opened by him" should be "He opened the door".
        ```

        If no significant style issues are found, return the single line: "No major style issues found."
        """
        return {"prompt": prompt}

    @staticmethod
    def enhance_narrative_flow(draft: str, outline: str, num_chapters: int) -> dict:
        prompt = f"""Analyze the narrative flow and pacing of the following draft ({num_chapters} chapters), considering the original outline.

        DRAFT:
        {draft}

        BOOK OUTLINE:
        {outline}

        Focus on:
        1. Chapter Transitions: Smoothness and logic between chapters.
        2. Scene Flow: Connections between scenes, abrupt jumps.
        3. Pacing: Effectiveness, slow/rushed sections.
        4. Exposition vs. Action Balance: Natural integration of exposition.
        5. Subplot Integration: Effective weaving into the main narrative.
        6. Structural Adherence: Alignment with the outline, impact of deviations.

        **Output Format:**
        Provide your feedback as a markdown list. Each item should represent a distinct piece of feedback.
        For each feedback item, specify:
        - **Chapter:** The chapter number(s) or transition the feedback applies to (e.g., "Chapter 4", "Transition 6-7", "Chapters 9-11", "General"). Use "General" if it applies broadly.
        - **Category:** "Flow"
        - **Priority:** High, Medium, Low (Estimate impact on reader experience).
        - **Suggestion:** A clear description of the flow/pacing issue and a suggestion for improvement.

        Example:
        ```markdown
        - **Chapter:** Transition 6-7
          **Category:** Flow
          **Priority:** High
          **Suggestion:** The transition between the cliffhanger at the end of Chapter 6 and the calm opening of Chapter 7 feels abrupt. Consider adding a brief bridging scene or adjusting the start of Chapter 7.

        - **Chapter:** Chapters 9-11
          **Category:** Flow
          **Priority:** Medium
          **Suggestion:** The pacing in these chapters feels slow due to extended internal monologues. Consider trimming some introspection or adding more external action/dialogue.
        ```

        If no significant flow/pacing issues are found, return the single line: "No major flow issues found."
        """
        return {"prompt": prompt}

    @staticmethod
    def format_book(draft: str) -> dict:
        prompt = f"""Format the following final draft manuscript for presentation.

        DRAFT:
        {draft}

        Apply standard manuscript formatting:
        1. Consistent Chapter Headings: Use a clear format like "# Chapter X".
        2. Section Breaks: Use appropriate markdown separators (like '---') for scene breaks if identifiable.
        3. Paragraph Spacing: Ensure reasonable spacing between paragraphs.
        4. Dialogue Formatting: Ensure dialogue is clearly formatted (e.g., using standard quotation marks).
        5. Basic Cleanup: Remove any obvious leftover artifacts or notes not part of the narrative.
        6. Table of Contents (Optional): If possible, generate a simple markdown Table of Contents listing the chapters at the beginning.

        Return the fully formatted manuscript in markdown. Do not add commentary, just the formatted text."""
        return {"prompt": prompt}

class BookTasks:
    """Task definitions for book generation using pydantic-ai agents"""

    def __init__(self):
        self.prompts = TaskPrompts()

    async def extract_elements(self, concept: str) -> dict:
        """Extract key elements from the initial concept"""
        return self.prompts.extract_elements(concept)

    async def develop_concept(self, concept: str, elements: dict = None, publishing_metadata: Optional[dict] = None) -> dict:
        """Develop and refine the initial book concept using extracted elements and market insights"""
        return self.prompts.develop_concept(concept, elements, publishing_metadata)

    async def generate_title(self, concept: str) -> dict:
        """Generate a book title"""
        return self.prompts.generate_title(concept)

    async def create_book_outline(self, refined_concept: str, num_chapters: int, publishing_metadata: Optional[dict] = None) -> dict:
        """Create a detailed book outline based on the concept and market insights"""
        return self.prompts.create_outline(refined_concept, num_chapters, publishing_metadata)

    async def build_world(self, outline: str, concept: str, publishing_metadata: Optional[dict] = None) -> dict:
        """Build the world/setting for the book incorporating market insights"""
        return self.prompts.build_world(outline, concept, publishing_metadata)

    async def develop_characters(self, outline: str, world: str, publishing_metadata: Optional[dict] = None) -> dict:
        """Develop detailed character profiles incorporating market insights"""
        return self.prompts.develop_characters(outline, world, publishing_metadata)

    async def write_chapter(
        self,
        outline: str,
        characters: List[CharacterProfile], # Use the Pydantic model from graph_state
        world: str,
        chapter_num: int,
        previous_chapter_content: str,
        min_words: int,
        state: BookGenerationState
    ) -> dict:
        """Write a single chapter with access to full state"""
        # Format characters for the prompt if needed, or pass structured data if agent supports
        # Simple string formatting for now:
        characters_str = "\n".join([
            f"Name: {c.name}\nRole: {c.role}\nDescription: {c.description}\nMotivations: {c.motivations}\n---"
            for c in characters
        ]) if characters else "No character details provided."

        return self.prompts.write_chapter(
            outline,
            characters_str, # Pass formatted string
            world,
            chapter_num,
            previous_chapter_content,
            min_words,
            state
        )

    async def review_book(self, content: str) -> dict:
        """Review the complete manuscript"""
        return self.prompts.review_book(content)

    async def polish_book(self, content: str, review: str) -> dict:
        """Polish the final manuscript based on review"""
        # This task might be deprecated or adapted if the new polishing flow replaces it.
        # Keeping it for now, but the PolishBook node uses the new agents.
        return self.prompts.polish_book(content, review)

    async def check_consistency(self, draft: str, characters: List[CharacterProfile], world: str, num_chapters: int) -> dict:
        """Check the draft for consistency."""
        # Format characters for the prompt
        characters_str = "\n".join([
            f"- Name: {c.name}\n  Role: {c.role}\n  Description: {c.description}\n  Motivations: {c.motivations}\n  Background: {getattr(c, 'background', 'Not specified')}"
            for c in characters
        ]) if characters else "No character details provided."
        return self.prompts.check_consistency(draft, characters_str, world, num_chapters)

    async def refine_style(self, draft: str, style_guide: Optional[StyleGuide], num_chapters: int) -> dict:
        """Analyze and suggest style improvements."""
        style_guide_str = style_guide.model_dump_json(indent=2) if style_guide else None
        return self.prompts.refine_style(draft, style_guide_str, num_chapters)

    async def enhance_narrative_flow(self, draft: str, outline: str, num_chapters: int) -> dict:
        """Analyze narrative flow and pacing."""
        return self.prompts.enhance_narrative_flow(draft, outline, num_chapters)

    async def format_book(self, draft: str) -> dict:
        """Apply final formatting to the book."""
        return self.prompts.format_book(draft)
