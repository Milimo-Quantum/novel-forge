from dotenv import load_dotenv
load_dotenv()

import logfire  
import streamlit as st
import re
import asyncio
import nest_asyncio
nest_asyncio.apply()  # Allow nested event loops
import os
import json
import time
import plotly.express as px
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

import streamlit_authenticator as stauth  # Added for authentication


# Assuming BookCrew now takes concept directly and uses graph
from book_crew import BookCrew
from utils import get_model_tags # Keep utility for fetching models

# Initialize Logfire monitoring
logfire.configure(
    project_name="novel-forge",
    send_to_logfire=True, # Set to False to disable Logfire
    token=os.getenv("LOGFIRE_TOKEN") # Ensure LOGFIRE_TOKEN is set in environment
)
# logfire.instrument_streamlit() # Removed: This function does not exist or is deprecated. Basic configure is sufficient.

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NovelForgeApp:
    # Class-level configuration (defaults can be overridden by BookCrew's config)
    DEFAULT_MODEL = BookCrew.DEFAULT_MODEL
    MIN_CHAPTERS = BookCrew.MIN_CHAPTERS
    MAX_CHAPTERS = BookCrew.MAX_CHAPTERS
    DEFAULT_CHAPTERS = BookCrew.DEFAULT_CHAPTERS
    DEFAULT_MIN_WORDS = BookCrew.DEFAULT_MIN_WORDS

    def __init__(self):
        """Initialize the NovelForge application."""
        self.setup_streamlit()

        # --- User Authentication ---
        try:
            with open("users.json", "r") as f:
                credentials = json.load(f)
        except Exception:
            credentials = {
                "usernames": {
                    "admin": {
                        "email": "admin@example.com",
                        "name": "Admin User",
                        "password": stauth.Hasher().hash("password123")
                    }
                }
            }

        # Always inject test user for convenience during development
        try:
            credentials.setdefault("usernames", {})
            credentials["usernames"]["Milimo"] = {
                "email": "milimo@example.com",
                "name": "Milimo",
                "password": stauth.Hasher().hash("Mainza123")
            }
        except Exception as e:
            logger.warning(f"Failed to inject test user: {e}")

        # Initialize nav_tab default if not set
        if 'nav_tab' not in st.session_state:
            st.session_state['nav_tab'] = 'Login/Signup'

        # Before sidebar: handle navigation redirect after login
        if st.session_state.get('navigate_to_project_config'):
            st.session_state['nav_tab'] = 'Project Config'
            st.session_state['navigate_to_project_config'] = False

        # Sidebar with animated logo (no navigation)
        with st.sidebar:
            logo_path = "assets/logo.png"
            if os.path.exists(logo_path):
                st.image(logo_path, use_container_width=True, output_format="PNG", caption="", channels="RGB", clamp=False, width=None)
            else:
                st.markdown("### ")

        # Authentication panel embedded or modal
        if 'authentication_status' not in st.session_state or st.session_state.get("authentication_status") is None or st.session_state.get("authentication_status") is False:
            tab = st.radio("Account", ["Login", "Sign Up"], horizontal=True)
            if tab == "Sign Up":
                st.subheader("Create a New Account")
                new_username = st.text_input("Username")
                new_email = st.text_input("Email")
                new_name = st.text_input("Full Name")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                if st.button("Register"):
                    if new_password != confirm_password:
                        st.error("Passwords do not match.")
                    elif new_username in credentials.get("usernames", {}):
                        st.error("Username already exists.")
                    elif not re.match(r"[^@]+@[^@]+\.[^@]+", new_email):
                        st.error("Invalid email address.")
                    else:
                        hashed_pw = stauth.Hasher().hash(new_password)
                        credentials.setdefault("usernames", {})[new_username] = {
                            "email": new_email,
                            "name": new_name,
                            "password": hashed_pw
                        }
                        with open("users.json", "w") as f:
                            json.dump(credentials, f, indent=2)
                        st.success("Registration successful! Please log in.")
                        st.stop()

            # Login tab
            authenticator = stauth.Authenticate(
                credentials,
                "novelforge_auth",
                "novelforge_auth_cookie",
                cookie_expiry_days=1
            )
            authenticator.login("main")

            auth_status = st.session_state.get("authentication_status")
            if auth_status is False:
                st.error("Username/password is incorrect")
                st.stop()
            elif auth_status is None:
                st.warning("Please enter your username and password")
                st.stop()
            else:
                st.success(f"Welcome {st.session_state.get('name', 'User')}!")
                # Set flag to navigate on next rerun
                st.session_state['navigate_to_project_config'] = True

        else:
            # If authenticated, proceed
            auth_status = st.session_state.get("authentication_status")
            if auth_status is not True:
                st.warning("Please login first via the Login/Signup tab.")
                st.stop()

        self.initialize_state()
        # _progress_placeholder is used to dynamically update the progress section
        self._progress_placeholder = st.empty()
        self.render_ui() # Render initial UI

    async def progress_handler(self, stage: str, message: str, agent_name: str, percent: int, stats: Optional[dict] = None):
        """Handle real-time progress updates from the BookCrew graph run."""
        logger.info(f"Progress Update: Stage='{stage}', Agent='{agent_name}', Percent={percent}%, Message='{message}'")

        # Update session state which triggers UI refresh
        st.session_state.progress = percent
        st.session_state.status = stage # Use the graph stage as the main status
        st.session_state.status_message = message # Store the specific message
        st.session_state.agent_name = agent_name # Often "GraphRunner" or specific agent if available

        # Update detailed stats if provided
        if stats:
            # Ensure nested dictionaries are handled correctly
            current_stats = st.session_state.stats
            current_stats.update({
                'current_content': stats.get('current_content', current_stats.get('current_content', '')),
                'completed_chapters': stats.get('completed_chapters', current_stats.get('completed_chapters', 0)),
                'total_chapters': stats.get('total_chapters', current_stats.get('total_chapters', 0)),
                'agent_metrics': stats.get('agent_metrics', current_stats.get('agent_metrics', {})),
                'chapter_metrics': stats.get('chapter_metrics', current_stats.get('chapter_metrics', [])),
                # Handle potential None for system_resources
                'system_resources': stats.get('system_resources') if stats.get('system_resources') else current_stats.get('system_resources', []),
                'words_per_minute': stats.get('words_per_minute', current_stats.get('words_per_minute', 0)),
                'words_generated': stats.get('words_generated', current_stats.get('words_generated', 0)),
                'elapsed_time': stats.get('elapsed_time', current_stats.get('elapsed_time', 0)),
                # Add other relevant stats from BookGenerationState if needed
            })
            st.session_state.stats = current_stats # Assign back to trigger update

        # Log to Logfire (optional, depends on desired granularity)
        log_message = f"Progress: {percent}% | Stage: {stage} | Agent: {agent_name} | Msg: {message}"
        # Pass extra data as keyword arguments
        logfire.info(log_message, stats_snapshot=stats)

        # No need to manually call render here, Streamlit handles it via state change

    def setup_streamlit(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Novel Forge",
            page_icon="",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        # Apply Custom CSS (keep existing styles)
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Exo:wght@400;700&display=swap');

        body, .main, .stApp {
            background: #f9f9f9;
            color: #222;
            font-family: 'Orbitron', 'Exo', sans-serif;
        }

        /* Sidebar */
        .css-1d391kg {
            background: #ffffff;
            border-right: 2px solid #4a90e2;
            backdrop-filter: blur(6px);
        }

        /* Inputs and buttons */
        .stButton > button, .stTextInput > div > div > input, .stNumberInput input, .stSlider > div, .stSelectbox > div, .stTextArea textarea {
            background-color: #fff;
            color: #222;
            border: 1px solid #4a90e2;
            border-radius: 8px;
            transition: all 0.3s ease;
        }

        /* Add spacing inside slider container to avoid overlap */
        .stSlider > div {
            padding: 0.5rem;
        }
        .stButton > button:hover {
            background-color: #4a90e2;
            color: #fff;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
        }

        /* Progress bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #4a90e2, #50e3c2);
            border-radius: 10px;
        }

        /* Panels */
        .main .block-container {
            background: #ffffff;
            border: 1px solid #ddd;
            border-radius: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            padding: 2rem;
            margin: 1rem auto;
        }

        /* Animated logo */
        .sidebar-logo {
            width: 100%;
            animation: spinLogo 10s linear infinite;
            margin-bottom: 1rem;
        }
        @keyframes spinLogo {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Tabs, cards, modals, extras */
        .stTabs, .stCard, .stModal, .stExpander {
            background: #ffffff;
            border: 1px solid #ddd;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        }

        /* Hover effects */
        .stButton > button:hover, .stDownloadButton > button:hover {
            box-shadow: 0 0 10px #4a90e2;
        }

        /* Syntax highlighted text areas */
        .stTextArea textarea {
            background: #fff;
            color: #222;
            font-family: 'Exo', monospace;
        }

        /* Status boxes */
        .status-box {
            padding: 1rem;
            border-radius: 15px;
            background: #f0f4f8;
            border: 1px solid #4a90e2;
            margin: 1rem 0;
        }

        /* Monitoring dashboard */
        .monitoring-dashboard {
            margin-top: 2rem;
            padding: 1rem;
            background: #f0f4f8;
            border-radius: 15px;
            border: 1px solid #4a90e2;
        }

        /* Footer */
        footer {
            color: #666;
            font-size: 0.8rem;
            text-align: center;
            margin-top: 2rem;
        }

        /* Scrollbars */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background: #4a90e2;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #50e3c2;
        }

        </style>
        """, unsafe_allow_html=True)

    def initialize_state(self):
        """Initialize or restore application state in st.session_state."""
        defaults = {
            'generation_started': False,
            'progress': 0,
            'status': "Ready",
            'status_message': "Waiting to start...",
            'agent_name': "N/A",
            'start_time': time.time(), # Initialize start time here
            'final_book_path': None,
            'final_stats_path': None,
            'error_message': None,
            'stats': { # Initialize stats structure based on progress_handler needs
                'current_content': '',
                'completed_chapters': 0,
                'chapters_approved': 0,
                'total_chapters': 0,
                'agent_metrics': {},
                'chapter_metrics': [],
                'system_resources': [], # Store the last snapshot dict
                'words_per_minute': 0,
                'average_wpm': 0,
                'words_generated': 0,
                'total_writing_time': 0,
                'elapsed_time': 0,
            }
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def render_header(self):
        """Render application header."""
        # Ensure assets directory exists
        os.makedirs("assets", exist_ok=True)
        col1, col2 = st.columns([1, 5])
        with col1:
            logo_path = "assets/logo.png"
            if os.path.exists(logo_path):
                try:
                    st.image(logo_path, width=120)
                except Exception as e:
                    logger.error(f"Failed to load logo '{logo_path}': {e}")
                    st.markdown("### ") # Fallback icon
            else:
                st.markdown("### ") # Fallback icon
                logger.warning(f"Logo file not found at '{logo_path}'")

        with col2:
            st.title("Novel Forge")
            st.markdown("""
            <div style='margin-top: -10px;'>
                <h4>AI Agent Authorship: Turning Ideas into Bestsellers</h4>
                <p style='color: #666;'>Powered by Milimo Quantum & Pydantic AI</p>
            </div>
            """, unsafe_allow_html=True)

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models including Ollama and OpenRouter."""
        try:
            # Use caching to avoid repeated calls during re-renders
            @st.cache_data(ttl=300)  # Cache for 5 minutes
            def _fetch_models():
                return get_model_tags()

            models = _fetch_models() or []

            # Add OpenRouter models statically
            openrouter_models = [
                {
                    "name": "meta-llama/llama-4-scout:free",
                    "details": {
                        "family": "OpenRouter",
                        "parameter_size": "Unknown",
                        "quantization_level": "N/A"
                    }
                }
            ]

            combined_models = models + openrouter_models

            if not combined_models:
                st.error("No models found. Ensure Ollama is running or OpenRouter API key is set.")
                return []

            return sorted(combined_models, key=lambda x: x.get("name", ""))
        except Exception as e:
            logger.error(f"Error getting models: {e}", exc_info=True)
            st.error(f"Failed to fetch models: {e}")
            return []

    def verify_directories(self) -> bool:
        """Verify required directories exist (delegated to BookCrew/Nodes now)."""
        # This can likely be removed as nodes/BookCrew handle directory creation
        logger.info("Directory verification responsibility moved to graph execution.")
        return True # Assume success, handled elsewhere

    # handle_file_upload can be removed if concept is always text area
    # def handle_file_upload(self) -> Optional[str]: ...

    def render_monitoring_dashboard(self):
        """Render monitoring dashboard using data from st.session_state.stats."""
        from streamlit_echarts import st_echarts

        stats = st.session_state.stats
        if not stats:
            return

        st.markdown("### Monitoring Dashboard")
        with st.container():
            st.markdown("<div class='monitoring-dashboard'>", unsafe_allow_html=True)

            # --- POV Distribution with ECharts ---
            chapter_metrics = stats.get('chapter_metrics', [])
            pov_counts = {}
            pov_colors = {}
            for chapter in chapter_metrics:
                pov = chapter.get('pov_character', 'Unknown')
                pov_counts[pov] = pov_counts.get(pov, 0) + 1
                color = chapter.get('color_code')
                if pov and color:
                    pov_colors[pov] = color

            if pov_counts:
                pov_data = [{"value": count, "name": pov, "itemStyle": {"color": pov_colors.get(pov, "#ccc")}} for pov, count in pov_counts.items()]
                pov_options = {
                    "title": {"text": "POV Character Distribution", "left": "center"},
                    "tooltip": {"trigger": "item"},
                    "legend": {"orient": "vertical", "left": "left"},
                    "series": [{
                        "name": "POV",
                        "type": "pie",
                        "radius": "50%",
                        "data": pov_data,
                        "emphasis": {
                            "itemStyle": {
                                "shadowBlur": 10,
                                "shadowOffsetX": 0,
                                "shadowColor": "rgba(0, 0, 0, 0.5)"
                            }
                        }
                    }]
                }
                st_echarts(options=pov_options, height="400px", key=f"echarts_pov_{time.time()}")

            # --- Agent Activity with ECharts ---
            agent_metrics = stats.get('agent_metrics', {})
            if agent_metrics:
                agent_names = list(agent_metrics.keys())
                times = [agent_metrics[a].get('total_time', 0) if isinstance(agent_metrics[a], dict) else 0 for a in agent_names]
                calls = [agent_metrics[a].get('call_count', 0) if isinstance(agent_metrics[a], dict) else 0 for a in agent_names]

                agent_options = {
                    "title": {"text": "Agent Activity", "left": "center"},
                    "tooltip": {"trigger": "axis"},
                    "legend": {"data": ["Time Spent (s)", "Call Count"], "top": 30},
                    "xAxis": {"type": "category", "data": agent_names},
                    "yAxis": [{"type": "value", "name": "Seconds"}, {"type": "value", "name": "Calls"}],
                    "series": [
                        {
                            "name": "Time Spent (s)",
                            "type": "bar",
                            "data": times,
                            "yAxisIndex": 0
                        },
                        {
                            "name": "Call Count",
                            "type": "line",
                            "data": calls,
                            "yAxisIndex": 1
                        }
                    ]
                }
                st_echarts(options=agent_options, height="400px", key=f"echarts_agents_{time.time()}")

            # --- Chapter Progress with ECharts ---
            chapter_data = []
            for chapter in chapter_metrics:
                chapter_data.append({
                    "chapter": chapter.get('chapter_number', 0),
                    "word_count": chapter.get('word_count', 0),
                    "status": chapter.get('status', 'unknown')
                })

            if chapter_data:
                x_data = [c["chapter"] for c in chapter_data]
                y_data = [c["word_count"] for c in chapter_data]

                chapter_options = {
                    "title": {"text": "Chapter Word Counts", "left": "center"},
                    "tooltip": {"trigger": "axis"},
                    "xAxis": {"type": "category", "data": x_data},
                    "yAxis": {"type": "value"},
                    "series": [{
                        "data": y_data,
                        "type": "line",
                        "smooth": True,
                        "areaStyle": {}
                    }]
                }
                st_echarts(options=chapter_options, height="400px", key=f"echarts_chapters_{time.time()}")

            # --- System Resources ---
            system_resources_snapshot = stats.get('system_resources')
            if system_resources_snapshot and isinstance(system_resources_snapshot, dict):
                st.markdown("#### System Resources (Last Snapshot)")
                st.metric("CPU %", f"{system_resources_snapshot.get('cpu_percent', 0):.1f}%")
                st.metric("Memory %", f"{system_resources_snapshot.get('memory_usage', 0):.1f}%")

            st.markdown("</div>", unsafe_allow_html=True)

    def render_stats(self):
        """Render key statistics using data from st.session_state.stats."""
        stats = st.session_state.stats
        if not stats: return

        # Calculate values with fallbacks from session state stats
        words_generated = stats.get('words_generated', 0)
        completed_chapters = stats.get('completed_chapters', 0)
        total_chapters = stats.get('total_chapters', 0)
        elapsed_seconds = stats.get('elapsed_time', 0)
        elapsed_minutes = elapsed_seconds / 60
        # WPM calculation might need refinement - based on total time vs active writing time
        wpm = stats.get('words_per_minute', 0) # Use value from state if available

        # Dynamically calculate WPM
        wpm = 0
        if elapsed_seconds > 0:
            wpm = (words_generated / elapsed_seconds) * 60
            stats['words_per_minute'] = wpm

        avg_wpm = stats.get('average_wpm', 0)
        approved_chapters = stats.get('chapters_approved', 0)
        total_writing_time = stats.get('total_writing_time', 0)

        st.markdown("### Writing Progress & Stats")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Words Generated", f"{words_generated:,}")
            st.metric("Words per Minute", f"{wpm:.1f}")
            st.metric("Average WPM", f"{avg_wpm:.1f}")

        with col2:
            st.metric("Completed Chapters", f"{completed_chapters}/{total_chapters}")

        with col3:
            st.metric("Elapsed Time (min)", f"{elapsed_minutes:.1f}")
            st.metric("Total Writing Time (min)", f"{total_writing_time/60:.1f}")
            st.metric("Total Words", f"{words_generated:,}")

    async def generate_book_async(self, concept: str, num_chapters: int, model_name: str, min_words: int, temperature: float, max_iterations: int, max_feedback_items: int):
        """Async function to run book generation, updating session state."""
        st.session_state.generation_started = True
        st.session_state.error_message = None # Clear previous errors
        st.session_state.final_book_path = None
        st.session_state.final_stats_path = None
        st.session_state.start_time = time.time() # Reset start time for this run

        # Reset stats for the new run
        self.initialize_state() # Re-initialize to clear old run data but keep defaults
        st.session_state.generation_started = True # Set again after reset
        st.session_state.stats['total_chapters'] = num_chapters # Set total chapters early

        try:
            crew = BookCrew(
                initial_concept=concept, # Pass concept directly
                num_chapters=num_chapters,
                model_name=model_name,
                progress_callback=self.progress_handler, # Pass the handler
                min_words_per_chapter=min_words,
                temperature=temperature,
            )
            # After BookCrew instantiation, update the config in the state with user values
            # (Assumes BookCrew exposes state/config as .state.config)
            if hasattr(crew, 'state') and hasattr(crew.state, 'config'):
                crew.state.config.max_iterations = max_iterations
                crew.state.config.max_feedback_items = max_feedback_items

            # Run the graph-based generation
            result_path_or_error = await crew.run()

            # Update session state with final results or error
            if crew.error_message:
                st.session_state.error_message = crew.error_message
                st.session_state.status = "Error"
                st.error(f"Generation failed: {crew.error_message}")
            else:
                st.session_state.final_book_path = crew.final_book_path
                st.session_state.final_stats_path = crew.final_stats_path
                st.session_state.status = "Complete"
                st.success("Book generation completed successfully!")

        except Exception as e:
            error_msg = f"Critical error during generation setup or execution: {e}"
            logger.error(error_msg, exc_info=True)
            st.session_state.error_message = error_msg
            st.session_state.status = "Error"
            st.error(error_msg)
        finally:
            st.session_state.generation_started = False # Mark generation as finished/stopped

    def render_ui(self):
        """Render the unified application UI."""
        self.render_header()

        # --- Project Configuration Form ---
        with st.expander(" Project Configuration", expanded=True):
            with st.form("book_generation_form"):
                st.markdown("### Book Concept")
                upload_tab, paste_tab = st.tabs(["Upload File", "Paste Text"])
                concept_from_file = None
                concept_from_text = None

                with upload_tab:
                    uploaded_file = st.file_uploader(
                        "Upload your book concept (.txt, .md)",
                        type=['txt', 'md'],
                        help="Upload a text file containing your book concept"
                    )
                    if uploaded_file:
                        try:
                            concept_from_file = uploaded_file.read().decode('utf-8')
                            if not concept_from_file.strip():
                                st.warning("Uploaded file is empty.")
                                concept_from_file = None
                        except Exception as e:
                            st.error(f"Error reading file: {e}")
                            concept_from_file = None

                with paste_tab:
                    concept_from_text = st.text_area(
                        "Paste your book concept",
                        height=200,
                        help="Paste or type your book concept here"
                    )

                concept = concept_from_file if concept_from_file else concept_from_text

                st.markdown("### Generation Parameters")
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    num_chapters = st.number_input(
                        "Number of Chapters", min_value=self.MIN_CHAPTERS, max_value=self.MAX_CHAPTERS,
                        value=self.DEFAULT_CHAPTERS, help="Choose the number of chapters"
                    )
                    min_words = st.number_input(
                        "Minimum Words per Chapter", min_value=100, max_value=5000,
                        value=self.DEFAULT_MIN_WORDS, step=50, help="Approximate minimum words per chapter"
                    )

                with col2:
                    available_models = self.get_available_models()
                    model_name = self.DEFAULT_MODEL
                    if available_models:
                        model_options = {
                            f"{m.get('name', 'Unknown')} ({m.get('details', {}).get('parameter_size', 'N/A')})": m.get('name')
                            for m in available_models if m.get('name')
                        }
                        default_index = 0
                        model_names_list = list(model_options.values())
                        if self.DEFAULT_MODEL in model_names_list:
                            default_index = model_names_list.index(self.DEFAULT_MODEL)

                        selected_option_display = st.selectbox(
                            "AI Model", options=list(model_options.keys()), index=default_index,
                            help="Choose the AI model"
                        )
                        model_name = model_options[selected_option_display]
                    else:
                        st.warning("No models found. Using default.")

                with col3:
                    st.markdown("###### Advanced")
                    temperature = st.slider(
                        "Creativity (Temp)", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                        help="Lower = predictable, Higher = creative"
                    )
                    max_iterations = st.number_input(
                        "Max Iterations per Node",
                        min_value=1, max_value=10, value=2, step=1,
                        help="Maximum number of iterations for each agent/node."
                    )
                    max_feedback_items = st.number_input(
                        "Max Feedback/Context Items",
                        min_value=1, max_value=20, value=6, step=1,
                        help="Maximum number of feedback/context items per agent/node."
                    )

                submit = st.form_submit_button(" Generate Book")

            if submit:
                if not concept or not concept.strip() or len(concept.strip()) < 10:
                    st.error("Please provide a meaningful book concept (at least 10 characters).")
                    st.stop()
                elif st.session_state.generation_started:
                    st.warning("Generation is already in progress.")
                else:
                    concept = concept.strip()
                    async def run_and_stream():
                        st.session_state.generation_started = True
                        st.session_state.error_message = None
                        st.session_state.final_book_path = None
                        st.session_state.final_stats_path = None
                        st.session_state.start_time = time.time()
                        self.initialize_state()
                        st.session_state.generation_started = True
                        st.session_state.stats['total_chapters'] = num_chapters

                        try:
                            async for stage, message, agent_name, percent, stats in self.generate_book_stream(
                                concept, num_chapters, model_name, min_words, temperature, max_iterations, max_feedback_items
                            ):
                                st.session_state.progress = percent
                                st.session_state.status = stage
                                st.session_state.status_message = message
                                st.session_state.agent_name = agent_name
                                if stats:
                                    st.session_state.stats.update(stats)
                                with self._progress_placeholder.container():
                                    st.markdown("---")
                                    st.markdown("### Generation Status")
                                    st.progress(percent / 100)
                                    status_color = "blue"
                                    if stage == "Complete": status_color = "green"
                                    elif stage == "Error": status_color = "red"
                                    st.markdown(
                                        f"""<div class='status-box'>
                                            <h4>Status: <span style='color:{status_color};'>{stage}</span></h4>
                                            <p><strong>Details:</strong> {message}</p>
                                            <p><strong>Agent/Task:</strong> {agent_name}</p>
                                        </div>""",
                                        unsafe_allow_html=True
                                    )
                                    current_content = st.session_state.stats.get('current_content', '')
                                    if 'agent_log' not in st.session_state:
                                        st.session_state['agent_log'] = []
                                    if current_content:
                                        if not st.session_state['agent_log'] or st.session_state['agent_log'][-1] != current_content:
                                            st.session_state['agent_log'].append(current_content)
                                        st.markdown("#### AI Writing (Live Stream)")
                                        unique_key = f"live_ai_output_{int(time.time() * 1000)}"
                                        st.text_area("Live AI Output", value=current_content, height=300, key=unique_key, disabled=True)
                                    self.render_stats()
                                    self.render_monitoring_dashboard()
                                    if stage == "Complete" and not st.session_state.error_message:
                                        st.success("Book generation completed successfully!")
                                    elif stage == "Error":
                                        st.error(f"Error: {st.session_state.error_message}")
                        except Exception as e:
                            error_msg = f"Critical error during generation: {e}"
                            logger.error(error_msg, exc_info=True)
                            st.session_state.error_message = error_msg
                            st.session_state.status = "Error"
                            st.error(error_msg)
                        finally:
                            st.session_state.generation_started = False

                    asyncio.run(run_and_stream())

        # --- Progress & Stats Section ---
        with self._progress_placeholder.container():
            st.markdown("### Generation Status")
            progress_value = st.session_state.get('progress', 0)
            try:
                progress_float = float(progress_value) / 100
                if progress_float < 0: progress_float = 0
                if progress_float > 1: progress_float = 1
            except (TypeError, ValueError):
                progress_float = 0
            st.progress(progress_float)

            status_color = "blue"
            if st.session_state.status == "Complete": status_color = "green"
            elif st.session_state.status == "Error": status_color = "red"
            st.markdown(
                f"""<div class='status-box'>
                    <h4>Status: <span style='color:{status_color};'>{st.session_state.status}</span></h4>
                    <p><strong>Details:</strong> {st.session_state.status_message}</p>
                    <p><strong>Agent/Task:</strong> {st.session_state.agent_name}</p>
                </div>""",
                unsafe_allow_html=True
            )

            if st.session_state.error_message:
                st.error(f"Error: {st.session_state.error_message}")

            self.render_stats()
            self.render_monitoring_dashboard()

        # --- Downloads Section ---
        if st.session_state.get('status') == "Complete" and not st.session_state.get('error_message'):
            st.markdown("### Download Results")
            col1, col2, col3, col4 = st.columns(4)
            book_path = st.session_state.get('final_book_path')
            stats_path = st.session_state.get('final_stats_path')

            if book_path and os.path.exists(book_path):
                with col1:
                    try:
                        with open(book_path, 'rb') as f:
                            st.download_button(
                                " Download Markdown",
                                data=f,
                                file_name=os.path.basename(book_path),
                                mime="text/markdown"
                            )
                    except Exception as e:
                        st.error(f"Error reading book file: {e}")
            else:
                with col1:
                    st.warning("Book file not found.")

            if stats_path and os.path.exists(stats_path):
                with col2:
                    try:
                        with open(stats_path, 'rb') as f:
                            st.download_button(
                                " Download Stats (JSON)",
                                data=f,
                                file_name=os.path.basename(stats_path),
                                mime="application/json"
                            )
                    except Exception as e:
                        st.error(f"Error reading stats file: {e}")
            else:
                with col2:
                    st.warning("Stats file not found.")

            # EPUB download
            epub_stats = st.session_state.get('stats', {})
            epub_path = epub_stats.get('epub_path') or st.session_state.get('epub_path')
            epub_status = epub_stats.get('epub_export_status', 'unknown')
            if epub_path and os.path.exists(epub_path):
                with col3:
                    try:
                        with open(epub_path, 'rb') as f:
                            st.download_button(
                                " Download EPUB",
                                data=f,
                                file_name=os.path.basename(epub_path),
                                mime="application/epub+zip"
                            )
                    except Exception as e:
                        st.error(f"Error reading EPUB file: {e}")
            else:
                with col3:
                    if epub_status == 'failed':
                        st.error("EPUB export failed.")
                    else:
                        st.info("EPUB file not found or not generated yet.")

            # PDF download
            pdf_path = epub_stats.get('pdf_path') or st.session_state.get('pdf_path')
            pdf_status = epub_stats.get('pdf_export_status', 'unknown')
            if pdf_path and os.path.exists(pdf_path):
                with col4:
                    try:
                        with open(pdf_path, 'rb') as f:
                            st.download_button(
                                " Download PDF",
                                data=f,
                                file_name=os.path.basename(pdf_path),
                                mime="application/pdf"
                            )
                    except Exception as e:
                        st.error(f"Error reading PDF file: {e}")
            else:
                with col4:
                    if pdf_status == 'failed':
                        st.error("PDF export failed.")
                    else:
                        st.info("PDF file not found or not generated yet.")

            # DOCX download
            docx_path = epub_stats.get('docx_path') or st.session_state.get('docx_path')
            docx_status = epub_stats.get('docx_export_status', 'unknown')
            if docx_path and os.path.exists(docx_path):
                st.download_button(
                    " Download DOCX",
                    data=open(docx_path, 'rb'),
                    file_name=os.path.basename(docx_path),
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            else:
                if docx_status == 'failed':
                    st.error("DOCX export failed.")
                else:
                    st.info("DOCX file not found or not generated yet.")

    async def generate_book_stream(self, concept, num_chapters, model_name, min_words, temperature, max_iterations, max_feedback_items):
        """Async generator yielding progress updates during book generation."""
        crew = BookCrew(
            initial_concept=concept,
            num_chapters=num_chapters,
            model_name=model_name,
            progress_callback=None,  # We'll handle progress manually
            min_words_per_chapter=min_words,
            temperature=temperature,
        )
        # After BookCrew instantiation, update the config in the state with user values
        # (Assumes BookCrew exposes state/config as .state.config)
        if hasattr(crew, 'state') and hasattr(crew.state, 'config'):
            crew.state.config.max_iterations = max_iterations
            crew.state.config.max_feedback_items = max_feedback_items

        async def progress_emitter(stage, message, agent_name, percent, stats):
            """Inner callback to yield progress info."""
            yield (stage, message, agent_name, percent, stats)

        queue = asyncio.Queue()

        async def callback(stage, message, agent_name, percent, stats):
            await queue.put((stage, message, agent_name, percent, stats))

        crew.progress_callback = callback

        # Start the graph run concurrently
        async def run_crew():
            try:
                await crew.run()
            except Exception as e:
                await queue.put(("Error", str(e), "System", 100, {}))
            finally:
                await queue.put(("Complete", "Generation finished", "System", 100, {}))

        task = asyncio.create_task(run_crew())

        while True:
            update = await queue.get()
            yield update
            if update[0] in ("Complete", "Error"):
                break

if __name__ == "__main__":
    app = NovelForgeApp()
