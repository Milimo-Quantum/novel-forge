# NovelForge AI Book Generator

NovelForge is a sophisticated AI authorship platform that orchestrates a **collaborative team of specialized AI agents** through a **persistent, feedback-driven graph workflow** to generate entire novels from a simple concept. It deeply integrates **pydantic-ai** for multi-agent coordination and **pydantic-graph** for modular, resumable orchestration, leveraging local LLMs via **Ollama** and cloud models via **OpenRouter**.

---

## Key Features

- **Persistent Graph Workflow:** Built with `pydantic_graph`, defining a detailed, checkpointed graph of 40+ nodes covering market analysis, concept development, outlining, worldbuilding, character creation, writing, review, editing, formatting, exporting, and reporting.
- **Explicit Multi-Agent Collaboration:** Uses `pydantic-ai` to instantiate dozens of specialized agents (Writer, Reviewer, Editor, World Builder, Dialogue Specialist, etc.) via the `BookAgents` factory, each an `EnhancedAgent` supporting explicit feedback channels.
- **Iterative Feedback Loops:** Agents exchange structured feedback explicitly (`process_with_feedback`), enabling multi-round refinement during writing and review phases.
- **Advanced Concept Propagation:** The initial concept is refined by a dedicated agent node into a canonical `refined_concept`, which is used by all downstream nodes and agent prompts. There are no arbitrary or restrictive integrity checks—creative evolution is fully agent-driven and robust.
- **Adaptive Model Selection:** Dynamically chooses optimal models (Ollama or OpenRouter) per agent role and task complexity, with caching and provider fallback.
- **Checkpointing & Resumability:** Uses `FileStatePersistence` to persist graph state and agent context after key nodes, supporting full recovery and incremental development.
- **Advanced Summarization & Reference Management:** Generates chapter and combined summaries, verifies and annotates cross-references with fuzzy matching and auto-updates.
- **Multi-Stage Review Pipeline:** Includes peer review simulation, editorial review, consistency checks, style refinement, flow enhancement, structural and line editing, coordinated via explicit graph nodes.
- **Quality Metrics & Heuristics:** Calculates readability scores (Flesch-Kincaid, SMOG, Coleman-Liau), dialogue ratio, description density, engagement, and style consistency to guide rewrites.
- **Publishing Preparation:** Generates front/back matter, metadata, and exports to Markdown, EPUB, PDF, DOCX.
- **Rich Streamlit UI:** Authentication, concept input, parameter tuning, live progress streaming, interactive charts, approval workflow, multi-format export.
- **Local & Cloud LLM Support:** Integrates with Ollama (local models) and OpenRouter (cloud models) via OpenAI-compatible APIs.
- **Logfire Integration:** Optional detailed monitoring and tracing.
- **Version Control:** Maintains full version history of state and chapters, supporting diffs, restores, and annotations.

---

## Workflow Details

### Concept Propagation & Refinement
- The workflow begins with an `initial_concept` provided by the user.
- A specialized agent node (DevelopConcept) refines this concept using advanced LLMs and agentic feedback, producing a `refined_concept`.
- The `refined_concept` becomes the canonical source of truth for all subsequent nodes, prompts, and agent tasks.
- If for any reason the refined concept is unavailable, the system safely falls back to the initial concept.
- There are no arbitrary or restrictive “concept integrity” checks: the workflow is fully agent-driven, empowering creative and meaningful evolution of the core idea.
- This approach ensures maximum flexibility, robustness, and creative power, strictly following the pydantic-ai agent framework.

1. **Start Generation:** Initialize state with concept and config.
2. **Generate Publishing Proposal:** Market analysis, audience, comparables, genre positioning.
3. **Generate Title:** Catchy book title.
4. **Develop Concept:** Refine and expand initial idea.
5. **Create Outline:** Chapter-by-chapter breakdown.
6. **Build World & Develop Characters:** Parallel generation with feedback refinement.
7. **Refine World & Characters:** Cross-inform details.
8. **Write Chapters:** Multi-agent writing (narrative, description, dialogue, continuity), iterative review/revision.
9. **Generate Summaries:** Chapter and combined summaries for context.
10. **Verify Cross-References:** Annotate, auto-update, and map references.
11. **Review Book:** Multi-stage review pipeline.
12. **Evaluate Quality Metrics:** Readability, genre heuristics, style, engagement.
13. **Generate Front & Back Matter:** Copyright, dedication, about author, acknowledgments.
14. **Assemble Book:** Combine all content.
15. **Polish Book:** Final editing.
16. **Format Book:** Consistent formatting.
17. **Save & Export:** Markdown, EPUB, PDF, DOCX, plus stats.

---

## User Interface Enhancements

- **Login System:** Secure access with username/password.
- **Drag-and-Drop Plot Board:** Reorder chapters interactively.
- **Editable Character Table:** Modify character profiles in a spreadsheet view.
- **Interactive Charts:** Visualize POV distribution, agent activity, chapter progress.
- **Export Options:** Download book in multiple formats.
- **Approval Workflow:** Approve/refine concept, outline, characters before generation.
- **Live Progress Streaming:** Real-time updates during generation.

---

## Agent Specializations

- **Concept Developer**
- **Master Story Architect (Outliner)**
- **World Weaver**
- **Character Alchemist**
- **Master Wordsmith (Writer)**
- **Dialogue Specialist**
- **Description Architect**
- **Plot Continuity Guardian**
- **Critical Reader (Reviewer)**
- **Literary Polisher (Editor)**
- **Prose Stylist**
- **Narrative Architect (Flow Enhancer)**
- **Formatting Specialist**

Agents exchange explicit feedback and are dynamically configured with adaptive model selection.

---

## Setup Instructions

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running
- Required Python packages:
  ```bash
  pip install -r requirements.txt
  pip install streamlit-authenticator streamlit-aggrid streamlit-echarts streamlit-sortables ebooklib fpdf python-docx markdown
  ```

### Model Setup

- Start Ollama:
  ```bash
  ollama serve
  ```
- Pull desired models:
  ```bash
  ollama pull llama3
  ollama pull granite3.2:8b-instruct-fp16
  ollama pull mistral:7b-instruct
  ollama pull granite3.2:8b-instruct-fp16
  ```
- Models are auto-selected per task.

### Run the App

```bash
streamlit run app.py
```

---

## Usage

- Login with your credentials.
- Upload or paste your book concept.
- Configure chapters, word count, temperature, and model.
- Approve/refine concept, outline, characters.
- Start generation.
- Monitor progress, review summaries, download results in multiple formats.

---

## Output

- **Drafts:** `novelForge/books/drafts/`
- **Chapters:** `novelForge/books/chapters/<run_id>/`
- **Run State:** `novelForge/runs/<run_id>.json`
- **Publishing Metadata:** Embedded in state and front/back matter.

---

## Advanced Features

- **Persistence:** Resume interrupted runs.
- **Monitoring:** Progress, agent activity, system resources.
- **Quality Assurance:** Readability, genre alignment, style, engagement.
- **Cross-References:** Annotated, auto-updated, and visualizable.
- **User Control:** Freeze flags, feedback, approval checkpoints.
- **Professional Publishing:** Proposal, front/back matter, multi-stage editing.
- **Adaptive Model Selection:** Chooses best model per task.
- **Export Formats:** Markdown, EPUB, PDF, DOCX, JSON stats.

---

## License

MIT License
