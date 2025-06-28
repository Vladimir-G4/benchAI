import streamlit as st
import os, json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from benchai.loader import load_test_cases
from benchai.runner import Runner
from benchai.visualizer import Visualizer
from benchai.types import UseCase
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from datetime import datetime
import time
from streamlit.components.v1 import html

st.set_page_config(page_title="benchAI - Model Evaluation Platform",
    page_icon="🔧", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for YC-inspired minimalist light theme
st.markdown("""
    <style>
    .main { 
        background-color: #ffffff; 
        color: #1a1a1a; 
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
        padding: 20px;
    }
    .stButton>button {
        background-color: #f0652f;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 500;
        font-size: 14px;
        transition: background-color 0.2s ease, transform 0.1s ease;
    }
    .stButton>button:hover {
        background-color: #d55528;
        transform: translateY(-1px);
    }
    .stButton>button:disabled {
        background-color: #e5e5e5;
        color: #999999;
    }
    .stFileUploader { 
        border: 1px solid #e5e5e5; 
        border-radius: 6px; 
        padding: 12px; 
        background-color: #fafafa; 
        transition: border-color 0.2s ease;
    }
    .stFileUploader:hover {
        border-color: #f0652f;
    }
    .stSelectbox { 
        background-color: #fafafa; 
        border: 1px solid #e5e5e5; 
        border-radius: 6px; 
        padding: 8px; 
        color: #1a1a1a; 
    }
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #fafafa;
        color: #1a1a1a;
        border-radius: 6px;
    }
    .result-card { 
        background-color: #ffffff; 
        border: 1px solid #f0f0f0; 
        border-radius: 8px; 
        padding: 16px; 
        margin-bottom: 16px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .success { color: #28a745; }
    .failure { color: #dc3545; }
    .badge {
        display: inline-block;
        background-color: #f0f0f0;
        color: #333333;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-right: 8px;
    }
    .sidebar .stButton>button { 
        width: 100%; 
        background-color: #f0652f;
        border: none;
    }
    .sidebar .stButton>button:hover {
        background-color: #d55528;
    }
    h1 { 
        color: #1a1a1a; 
        font-weight: 600; 
        font-size: 30px; 
        margin-bottom: 10px;
    }
    h3 { 
        color: #333333; 
        font-weight: 500; 
        font-size: 18px; 
        margin-bottom: 12px;
    }
    .stSpinner { 
        display: flex; 
        justify-content: center; 
    }
    .stMarkdown, .stMarkdown p, .stMarkdown div { 
        color: #1a1a1a; 
        font-size: 14px;
    }
    .stSidebar { 
        background-color: #f5f5f5; 
        padding: 20px;
    }
    .stProgress .st-bo {
        background-color: #f0652f;
    }
    .score-card {
        background-color: #ffffff;
        border: 1px solid #f0f0f0;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }
    .score-gauge {
        font-size: 36px;
        font-weight: 700;
        color: #1a1a1a;
    }
    .score-label {
        font-size: 16px;
        color: #666666;
    }
    </style>
""", unsafe_allow_html=True)

def calculate_benchai_score(results: list) -> float:
    """
    Calculate the benchAI Score based on weighted average of test case scores,
    adjusted by difficulty (easy=1.0, medium=1.5, hard=2.0).
    
    Args:
        results: List of EvalResult objects from benchAI runner.
    
    Returns:
        Float representing the benchAI Score (0-100).
    """
    if not results:
        return 0.0
    
    difficulty_weights = {"easy": 1.0, "medium": 1.5, "hard": 2.0}
    total_weighted_score = 0.0
    total_weight = 0.0
    
    for result in results:
        # Extract difficulty from test case metadata (if available)
        difficulty = result.metadata.get("difficulty", "medium") if hasattr(result, "metadata") else "medium"
        weight = difficulty_weights.get(difficulty, 1.5)
        total_weighted_score += result.score * weight
        total_weight += weight
    
    return (total_weighted_score / total_weight) * 100 if total_weight > 0 else 0.0

# Sidebar for navigation and settings
with st.sidebar:
    st.header("🔧 benchAI")
    st.markdown("Evaluate any LLM with precision. Use built-in test suites or upload your own.")
    
    # Select test source
    test_source = st.radio(
        "Test Source", 
        ["Built-in Tests", "Upload Custom Test File"], 
        help="Choose between benchAI's pre-built test suites or your own YAML/JSON file."
    )
    
    # Built-in test files from tests/fixtures/
    test_files = {
        "QA Tests": "tests/fixtures/qa_tests.yaml",
        "Summarization Tests": "tests/fixtures/summarization_tests.yaml"
    }
    
    selected_test = None
    uploaded_test_file = None
    if test_source == "Built-in Tests":
        selected_test = st.selectbox(
            "Select Test Suite", 
            list(test_files.keys()), 
            help="Pick a pre-built test suite for QA or Summarization."
        )
    else:
        uploaded_test_file = st.file_uploader(
            "Upload Test File (.yaml or .json)", 
            type=["yaml", "yml", "json"],
            help="Upload a YAML or JSON file with test cases (e.g., prompts, expected outputs)."
        )
    
    # Select use case
    use_case = st.selectbox(
        "Select Use Case", 
        [u.value for u in UseCase], 
        help="Select the task type (e.g., QA, Summarization) for evaluation."
    )
    
    # Upload model file
    model_file = st.file_uploader(
        "Upload Model File (.py)", 
        type=["py"], 
        help="Upload a Python file with a `model(prompt: str) -> str` function to evaluate."
    )
    
    # Run button
    run_evaluation = st.button("Run Evaluation", disabled=not (model_file and (selected_test or uploaded_test_file)))

# Main content
st.title("🔧benchAI - Model Evaluation Platform")
st.markdown("Test your AI model with a robust, model-agnostic platform built for real-world tasks.")

if run_evaluation:
    # Save uploaded files
    if uploaded_test_file:
        test_file_path = "uploaded_test_file.yaml"
        with open(test_file_path, "wb") as f:
            f.write(uploaded_test_file.read())
    else:
        test_file_path = test_files[selected_test]
    
    with open("uploaded_model.py", "wb") as f:
        f.write(model_file.read())
    
    # Load model
    try:
        spec = spec_from_file_location("uploaded_model", "uploaded_model.py")
        model_module = module_from_spec(spec)
        spec.loader.exec_module(model_module)
        
        if not hasattr(model_module, "model"):
            st.error("❌ Model file must define a function named `model(prompt: str) -> str`.", icon="🚨")
        else:
            model_fn = model_module.model
            with st.spinner("Evaluating model..."):
                try:
                    test_cases = load_test_cases(test_file_path)
                    test_cases = [tc for tc in test_cases if tc.use_case == UseCase(use_case)]
                    
                    if not test_cases:
                        st.error(f"❌ No test cases found for use case '{use_case}' in {test_file_path}.", icon="🚨")
                    else:
                        # Create the placeholder for terminal log
                        log_placeholder = st.empty()
                        log_buffer = []

                        def append_log(line: str):
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            log_buffer.append(f"[{timestamp}] {line}")
                            full_log = "\n".join(log_buffer)
                            log_placeholder.code(full_log, language="bash")
                            time.sleep(0.5)

                        # Simulate live terminal log
                        append_log("🔧 benchAI initialized...")
                        append_log(f"📤 Loaded {len(test_cases)} test cases from: {Path(test_file_path).name}")
                        append_log("🤖 Running model evaluation...")
                        
                        # Run evaluation
                        runner = Runner(model_fn)
                        results = []
                        for i, result in enumerate(runner.run(test_cases)):
                            results.append(result)
                        
                        append_log("✅ All test cases processed successfully")
                        append_log("📊 Evaluation complete! Generating visualizations and metrics...")

                        benchai_score = calculate_benchai_score(results)
                        score_percent = min(max(benchai_score, 0), 100)
                        score_color = "#28a745" if score_percent >= 80 else "#f4c542" if score_percent >= 50 else "#dc3545"
                        performance_label = "Excellent" if score_percent >= 80 else "Moderate" if score_percent >= 50 else "Needs Improvement"

                        html(f"""
                        <div style="
                            background-color: #ffffff;
                            border: 1px solid #e6e6e6;
                            border-radius: 12px;
                            padding: 32px;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
                            text-align: center;
                            max-width: 500px;
                            margin: 0 auto 32px auto;
                            font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
                        ">
                            <div style="font-size: 18px; color: #666; margin-bottom: 12px;">Overall Model Performance</div>
                            
                            <div style="position: relative; display: inline-block; width: 200px; height: 200px;">
                                <svg width="200" height="200">
                                    <circle cx="100" cy="100" r="80" stroke="#f0f0f0" stroke-width="16" fill="none" />
                                    <circle cx="100" cy="100" r="80" stroke="{score_color}" stroke-width="16" fill="none"
                                        stroke-dasharray="{int(score_percent*5.026)} 999"
                                        stroke-linecap="round"
                                        transform="rotate(-90 100 100)" />
                                    <text x="100" y="110" font-size="36" fill="#1a1a1a" font-weight="600" text-anchor="middle">{score_percent:.1f}</text>
                                </svg>
                            </div>
                            
                            <div style="margin-top: 12px; font-size: 16px; font-weight: 500; color: {score_color};">
                                benchAI Score – {performance_label}
                            </div>
                            
                            <p style="font-size: 14px; color: #444; margin-top: 4px;">
                                Weighted average across test case outcomes, normalized by difficulty level (easy, medium, hard).
                            </p>
                        </div>
                        """, height=320)
                        
                        # Export results
                        export_file = f"benchai_results_{use_case}_{Path(test_file_path).stem}.json"
                        with open(export_file, "w") as f:
                            json.dump([r.__dict__ for r in results], f, indent=2)
                        with open(export_file, "rb") as f:
                            st.download_button(
                                label="📥 Download Results (JSON)",
                                data=f,
                                file_name=export_file,
                                mime="application/json",
                                help="Export evaluation results as JSON for CI/CD or further analysis."
                            )
                        
                       # Display results
                        st.header("Evaluation Results")

                        # Gather metadata values for filters
                        all_difficulties = sorted(set(getattr(r, "metadata", {}).get("difficulty", "N/A") for r in results))
                        all_categories = sorted(set(getattr(r, "metadata", {}).get("category", "N/A") for r in results))
                        all_eval_types = sorted(set(getattr(r, "metadata", {}).get("eval_type", "N/A") for r in results))

                        # UI filter widgets (world-class)
                        with st.container():
                            st.markdown("""
                            <style>
                                .filter-row {
                                    display: flex;
                                    gap: 1.5rem;
                                    flex-wrap: wrap;
                                    margin-bottom: 20px;
                                }
                                .filter-label {
                                    font-weight: 600;
                                    margin-bottom: 4px;
                                    display: block;
                                    color: #333;
                                }
                            </style>
                            """, unsafe_allow_html=True)

                            st.markdown('<div class="filter-row">', unsafe_allow_html=True)
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                difficulty_filter = st.multiselect(
                                    "🎯 Difficulty",
                                    options=all_difficulties,
                                    default=all_difficulties,
                                    help="Filter test cases by difficulty level."
                                )
                            with col2:
                                category_filter = st.multiselect(
                                    "📂 Category",
                                    options=all_categories,
                                    default=all_categories,
                                    help="Filter by the test's logical grouping."
                                )
                            with col3:
                                eval_type_filter = st.multiselect(
                                    "🧪 Eval Type",
                                    options=all_eval_types,
                                    default=all_eval_types,
                                    help="Filter by the evaluation strategy (e.g., exact match, ROUGE, embedding)."
                                )
                            st.markdown('</div>', unsafe_allow_html=True)

                        # Filter results
                        filtered_results = [
                            r for r in results
                            if getattr(r, "metadata", {}).get("difficulty", "N/A") in difficulty_filter
                            and getattr(r, "metadata", {}).get("category", "N/A") in category_filter
                            and getattr(r, "metadata", {}).get("eval_type", "N/A") in eval_type_filter
                        ]

                        # Render filtered results
                        if filtered_results:
                            for i, r in enumerate(filtered_results):
                                metadata = getattr(r, "metadata", {})
                                difficulty = metadata.get("difficulty", "N/A")
                                category = metadata.get("category", "N/A")
                                eval_type = metadata.get("eval_type", "N/A")
                                with st.expander(f"Test Case {i+1}: {r.prompt[:60]}{'...' if len(r.prompt) > 60 else ''}", expanded=False):
                                    st.markdown(f"""
                                        <div class='result-card'>
                                            <div>
                                                <span class='badge' title='Test case category'>{category}</span>
                                                <span class='badge' title='Difficulty level'>{difficulty}</span>
                                                <span class='badge' title='Evaluation method'>{eval_type}</span>
                                            </div>
                                            <strong>Prompt:</strong> {r.prompt}<br>
                                            <strong>Expected:</strong> {r.expected}<br>
                                            <strong>Actual:</strong> {r.actual}<br>
                                            <strong>Score:</strong> {r.score:.2f} <span class='{'success' if r.passed else 'failure'}'>{'✅' if r.passed else '❌'}</span><br>
                                            <strong>Feedback:</strong> <i>{r.feedback}</i>
                                        </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.warning("No results match your current filter selection.")

                        
                        # Enhanced Visual Summary
                        st.header("Visual Summary")
                        
                        # Score Distribution Histogram
                        scores = [r.score for r in results]
                        fig_hist = px.histogram(
                            x=scores,
                            nbins=20,
                            title="Score Distribution",
                            labels={"x": "Score", "y": "Count"},
                            color_discrete_sequence=["#f0652f"],
                            template="plotly_white"
                        )
                        fig_hist.update_layout(
                            title_font_size=18,
                            xaxis_title_font_size=14,
                            yaxis_title_font_size=14,
                            showlegend=False,
                            margin=dict(t=50, b=50)
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Pass/Fail Pie Chart
                        pass_fail = pd.DataFrame({
                            "Status": ["Pass" if r.passed else "Fail" for r in results]
                        })
                        fig_pie = px.pie(
                            pass_fail,
                            names="Status",
                            title="Pass/Fail Breakdown",
                            color="Status",
                            color_discrete_map={"Pass": "#28a745", "Fail": "#dc3545"},
                            template="plotly_white"
                        )
                        fig_pie.update_layout(
                            title_font_size=18,
                            legend=dict(orientation="h", y=-0.1),
                            margin=dict(t=50, b=50)
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                        # Category Performance Bar Chart
                        categories = [getattr(r, "metadata", {}).get("category", "Unknown") for r in results]
                        category_scores = pd.DataFrame({
                            "Category": categories,
                            "Score": [r.score for r in results]
                        }).groupby("Category").mean().reset_index()
                        fig_bar = px.bar(
                            category_scores,
                            x="Category",
                            y="Score",
                            title="Performance by Category",
                            color="Score",
                            color_continuous_scale=["#dc3545", "#f0652f", "#28a745"],
                            template="plotly_white"
                        )
                        fig_bar.update_layout(
                            title_font_size=18,
                            xaxis_title_font_size=14,
                            yaxis_title_font_size=14,
                            yaxis_range=[0, 1],
                            margin=dict(t=50, b=50)
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Download Visuals
                        st.download_button(
                            label="📊 Download Visuals (PNG)",
                            data=fig_hist.to_image(format="png"),
                            file_name=f"benchai_visuals_{use_case}_{Path(test_file_path).stem}.png",
                            mime="image/png",
                            help="Download score distribution chart as PNG."
                        )
                        
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {str(e)}", icon="🚨")
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}", icon="🚨")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by the benchAI team. Model-agnostic, use-case-aware, CI/CD-ready.")