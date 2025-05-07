# Novel Forge Architecture Overview

Novel Forge is a **multi-agent, graph-driven AI authorship platform** built on the **pydantic-ai** and **pydantic-graph** frameworks. It orchestrates a complex workflow of specialized agents to generate, refine, review, and export full-length novels with adaptive feedback and iterative improvement.

---

## Core Components and Data Flow

### 1. **User Interface (app.py)**

- Built with **Streamlit**, providing:
  - User authentication and project setup
  - Input of book concept, parameters, and approvals
  - Real-time progress updates, live AI output, and monitoring dashboards
  - Download links for generated drafts and exports
- Initiates the generation process by creating a **BookCrew** instance.

### 2. **BookCrew Orchestration (book_crew.py)**

- Manages the **entire lifecycle** of book generation.
- Initializes:
  - **BookAgents**: container for all specialized agents
  - **BookTasks**: prompt templates and task functions
  - **BookGenerationState**: central state object
  - **pydantic-graph**: the execution graph of nodes
  - **FileStatePersistence**: for checkpointing and resuming
- Handles:
  - **Adaptive model selection** per agent role and task complexity
  - **Progress streaming** with detailed metrics
  - **Checkpoint saving** at key stages
  - **Resuming** from saved checkpoints
- Runs the graph **asynchronously**, updating UI in real time.

### 3. **Agents (agents.py)**

- Defines **EnhancedAgent** subclasses with:
  - Explicit feedback channels
  - Specialization metadata
  - `process_with_feedback()` method for structured feedback integration
- **BookAgents** factory creates:
  - Writer, Concept Developer, World Builder, Character Developer
  - Dialogue Specialist, Description Architect, Reviewers, Editors
  - Market Analyst, Genre Specialist, Exporters, and more
- Supports **adaptive model switching** (Ollama, OpenRouter) based on task.
- Integrates **feedback loops** for iterative refinement.

### 4. **Graph Nodes (graph_nodes.py)**

- Implements a **finite state machine** with pydantic-graph:
  - Nodes for concept development, outlining, world-building, character creation
  - Multi-agent coordination and iterative chapter writing
  - Multi-stage review, polishing, formatting, exporting
- **Explicit iteration control**:
  - Iterative refinement loops with convergence checks
  - Feedback capping and prioritization
  - Early stopping to prevent infinite loops
- **Feedback flows**:
  - Structured feedback passed between agents
  - Review feedback triggers rewrites or escalations
  - Polishing feedback integrated before final export
- **Persistence**:
  - Saves intermediate outputs, feedback, and checkpoints
  - Supports resuming from any saved state

### 5. **State and Models (graph_state.py)**

- **BookGenerationState** tracks:
  - Initial concept, config, outline, world, characters
  - All chapter versions, statuses, and feedback
  - Publishing metadata, summaries, combined drafts
  - Feedback history, applied changes, style guidelines
  - Metrics, agent activity, system resources
  - Version control with save/restore/diff capabilities
- **ChapterResult** stores multiple versions with feedback and metadata.
- **PolishingFeedback** structures feedback with priority and category.

### 6. **Task Prompts (tasks.py)**

- Centralizes **prompt engineering** for all stages:
  - Market analysis, genre, audience, titles
  - Outlining, world-building, character development
  - Chapter writing with detailed constraints
  - Review, polishing, consistency, style, flow
  - Formatting, metadata, export
- **BookTasks** wraps these prompts for async agent calls.

### 7. **Utilities (utils.py)**

- Loads style prompts (Paul Graham style)
- Manages model info, readiness, cleanup
- Provides logging and system monitoring

---

## Additional Architectural Features

- **Adaptive Iteration Control**: Iterative loops in chapter writing and coordination use **convergence checks** on content changes and **feedback prioritization** to stop early, preventing infinite or redundant cycles.
- **Structured Feedback Integration**: Feedback is accumulated as **PolishingFeedback** objects with priority, category, and before/after context, then **filtered and capped** before each iteration.
- **Versioning and Checkpointing**: The system saves **version snapshots** of the entire state, supports **diffs and restores**, and uses **FileStatePersistence** to resume from any checkpoint.
- **Adaptive Model Selection**: Models are dynamically chosen based on **task type, content length, and complexity**, optimizing cost and quality.
- **User Control Flags**: Users can **freeze** parts of the workflow (concept, outline, characters, world) and inject feedback to guide generation.
- **Style Prompt Injection**: Paul Graham style instructions are loaded and prepended to prompts to enforce consistent narrative style.
- **Real-time Monitoring and Analytics**: Tracks agent activity, system resources, writing stats, and streams detailed progress to the UI.

---

## Updated Architecture Diagram

*(Diagram retained from previous version for clarity)*

```mermaid
flowchart TD
    subgraph UI["Streamlit UI - app.py"]
        A1[User Inputs:<br>Concept, Params, Approvals]
        A2[NovelForgeApp]
        A3[Progress Updates<br>& Visualization]
        A4[User Auth<br>streamlit_authenticator]
        A5[Logfire Monitoring]
        A1 --> A2
        A4 --> A2
        A2 --> A3
        A2 --> A5
    end

    subgraph Crew["BookCrew - book_crew.py"]
        C1[Initialize BookCrew]
        C2[Setup Agents<br>BookAgents]
        C3[Setup Tasks<br>BookTasks]
        C4[Setup Graph<br>pydantic_graph]
        C5[Create State<br>BookGenerationState]
        C6[Run Graph Async<br>asyncio]
        C7[Progress Callback]
        C8[FileStatePersistence<br>checkpoint/resume]
        C9[Save Outputs:<br>Markdown, JSON, EPUB, PDF, DOCX]
        C10[Adaptive Model Selection]
        C1 --> C2
        C1 --> C3
        C1 --> C4
        C1 --> C5
        C1 --> C10
        C6 --> C7
        C6 --> C8
        C6 --> C9
        C2 --> C4
        C3 --> C4
        C5 --> C4
    end

    subgraph Agents["Agents - agents.py"]
        AG1[BookAgents Factory]
        AG2[EnhancedAgent<br>pydantic-ai]
        AG3[Model Selection<br>Ollama, OpenRouter]
        AG4[BookResult<br>progress tracking]
        AGS[Specialized Agents]
        AGS1[Writer]
        AGS2[Concept Developer]
        AGS3[World Builder]
        AGS4[Character Developer]
        AGS5[Dialogue Specialist]
        AGS6[Description Architect]
        AGS7[Plot Continuity Guardian]
        AGS8[Market Analyst]
        AGS9[Genre Specialist]
        AGS10[Audience Analyst]
        AGS11[Comparative Titles Researcher]
        AGS12[Strategic Outliner]
        AGS13[Plot Architect]
        AGS14[Write Coordinator]
        AGS15[Structural Editor]
        AGS16[Line Editor]
        AGS17[Peer Review Simulator]
        AGS18[Style Guide Enforcer]
        AGS19[Formatting Optimizer]
        AGS20[Metadata Specialist]
        AGS21[Exporter]
        AG1 --> AGS
        AGS --> AGS1 & AGS2 & AGS3 & AGS4 & AGS5 & AGS6 & AGS7 & AGS8 & AGS9 & AGS10 & AGS11 & AGS12 & AGS13 & AGS14 & AGS15 & AGS16 & AGS17 & AGS18 & AGS19 & AGS20 & AGS21
        AG1 --> AG2
        AG1 --> AG3
        AG2 --> AG4
        AG2 -.->|Feedback| AG2
    end

    subgraph Graph["Book Generation Graph - graph_nodes.py"]
        G00[StartGeneration]
        G01[MarketAnalysis]
        G02[GenrePositioning]
        G03[AudienceTargeting]
        G04[ComparativeTitles]
        G05[StrategicOutlineGenerator]
        G06[GenerateProposal]
        G1[GenerateTitle]
        G2[DevelopConcept]
        G3[CreateOutline]
        G4[BuildWorld]
        G5[DevelopCharacters]
        G6[RefineWorldWithCharacters]
        G7[RefineCharactersWithWorld]
        G8[WriteCoordinator]
        G9[WriteChapter<br>loop]
        G10[VerifyCrossReferences]
        G11[GenerateSummaries]
        G12[StructuralEditor]
        G13[LineEditor]
        G14[PeerReviewSimulator]
        G15[MultiStageReviewHub]
        G16[EvaluateQualityMetrics]
        G17[StyleGuideEnforcer]
        G18[FormattingOptimizer]
        G19[MarketMetadataGenerator]
        G20[PlatformExporter]
        G21[GenerateFrontMatter]
        G22[GenerateBackMatter]
        G23[ReviewBook]
        G24[AssembleBook]
        G25[PolishBook]
        G26[FormatBook]
        G27[SaveFinalBook]
        G28[End]

        G29[PeerReview]
        G30[EditorialReview]
        G31[ConsistencyCheck]
        G32[StyleRefinement]
        G33[FlowEnhancement]
        G34[ReviewAggregator]

        G35[ExportEPUB]
        G36[ExportPDF]
        G37[ExportDOCX]
        G38[FinalReport]

        G00 --> G01 --> G02 --> G03 --> G04 --> G05 --> G06 --> G1 --> G2 --> G3 --> G4 --> G5 --> G6 --> G7 --> G8 --> G9
        G9 --> G10 --> G11 --> G12 --> G13 --> G14 --> G15 --> G16 --> G17 --> G18 --> G19 --> G20 --> G21 --> G22 --> G23 --> G24 --> G25 --> G26 --> G27 --> G35 --> G36 --> G37 --> G38 --> G28

        G23 -- rewrites needed --> G9

        G23 --> G29 --> G34
        G23 --> G30 --> G34
        G23 --> G31 --> G34
        G23 --> G32 --> G34
        G23 --> G33 --> G34
        G34 --> G23

        G2 --> C8
        G3 --> C8
        G4 --> C8
        G5 --> C8
        G9 --> C8
        G23 --> C8
        G27 --> C8
    end

    subgraph State["State & Models - graph_state.py"]
        S1[BookGenerationState]
        S2[ChapterResult<br>multiple versions]
        S3[CharacterProfile]
        S4[WritingStats]
        S5[AgentMetrics]
        S6[PolishingFeedback]
        S7[StyleGuide]
        S1 --> S2
        S1 --> S3
        S1 --> S4
        S1 --> S5
        S1 --> S6
        S1 --> S7
    end

    subgraph Tasks["Task Prompts - tasks.py"]
        T1[TaskPrompts]
        T2[BookTasks]
        T2 --> T1
    end

    subgraph Utils["Utilities - utils.py"]
        U1[Model Tag Fetching]
        U2[Logging Helpers]
    end

    A2 --> C1
    C7 --> A3
    C2 --> AG1
    C3 --> T2
    C4 --> G00
    G9 --> AG2
    G01 --> T2
    G02 --> T2
    G03 --> T2
    G04 --> T2
    G05 --> T2
    G06 --> T2
    G1 --> T2
    G2 --> T2
    G3 --> T2
    G4 --> T2
    G5 --> T2
    G6 --> T2
    G7 --> T2
    G8 --> T2
    G9 --> T2
    G10 --> T2
    G11 --> T2
    G12 --> T2
    G13 --> T2
    G14 --> T2
    G15 --> T2
    G16 --> T2
    G17 --> T2
    G18 --> T2
    G19 --> T2
    G20 --> T2
    G21 --> T2
    G22 --> T2
    G23 --> T2
    G24 --> T2
    G25 --> T2
    G26 --> T2
    G27 --> T2

    G00 --> S1
    G9 --> S1
    G23 --> S1
    G27 --> S1
    C5 --> S1
    C5 --> C8
    C5 --> C9
    C2 --> U1
    C2 --> U2
