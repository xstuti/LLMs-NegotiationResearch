import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Ultimatum Game Visualizer",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .stApp {
        max-width: 100%;
    }
    .conversation-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        background-color: #ffffff;
    }
    .player-red {
        background-color: #ffe6e6;
        border-left: 4px solid #cc0000;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        color: #000000;
    }
    .player-blue {
        background-color: #e6f2ff;
        border-left: 4px solid #0066cc;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        color: #000000;
    }
    .game-marker {
        background-color: #f5f5f5;
        border-left: 4px solid #666666;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        color: #000000;
    }
    .stat-item {
        padding: 8px 0;
        border-bottom: 1px solid #eee;
        font-family: monospace;
    }
    .stat-label {
        font-weight: bold;
        color: #333;
    }
    .stat-value {
        color: #666;
        margin-left: 10px;
    }
    .combination-option {
        padding: 10px;
        margin: 5px 0;
        border: 2px solid #ddd;
        border-radius: 5px;
        cursor: pointer;
        background-color: white;
        transition: all 0.2s;
    }
    .combination-option:hover {
        border-color: #4CAF50;
        background-color: #f0f0f0;
    }
    .combination-option.selected {
        border-color: #4CAF50;
        background-color: #e8f5e9;
    }
    .iteration-buttons {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin: 15px 0;
    }
    .iteration-btn {
        min-width: 45px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_summary_data(log_dir):
    """Load summary.json from the log directory"""
    summary_path = Path(log_dir) / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            return json.load(f)
    return None


def get_combinations(log_dir):
    """Get all unique model vs behavior combinations"""
    summary_data = load_summary_data(log_dir)
    if not summary_data:
        return []

    combinations = []
    for key, value in summary_data.items():
        # Parse the key: "Model1_vs_Model2_Behavior"
        parts = key.split("_")
        if "vs" in parts:
            vs_idx = parts.index("vs")
            model1 = "_".join(parts[:vs_idx])
            model2 = "_".join(parts[vs_idx + 1 : -1])
            behavior = parts[-1]

            combinations.append(
                {
                    "key": key,
                    "model1": model1,
                    "model2": model2,
                    "behavior": behavior,
                }
            )

    return sorted(combinations, key=lambda x: (x["model1"], x["model2"], x["behavior"]))


def get_iteration_dirs(log_dir, combination_key):
    """Get all iteration directories for a specific combination"""
    base_name = combination_key.replace("_vs_", "_vs_")
    iteration_dirs = []

    log_path = Path(log_dir)
    for item in log_path.iterdir():
        if item.is_dir() and item.name.startswith(base_name) and "iter_" in item.name:
            # Extract iteration number
            iter_num = int(item.name.split("iter_")[-1])
            iteration_dirs.append({"path": item, "iteration": iter_num})

    return sorted(iteration_dirs, key=lambda x: x["iteration"])


def load_game_state(iteration_dir):
    """Load game_state.json from an iteration directory"""
    # Find the timestamp subdirectory
    for subdir in Path(iteration_dir).iterdir():
        if subdir.is_dir():
            game_state_path = subdir / "game_state.json"
            if game_state_path.exists():
                with open(game_state_path, "r") as f:
                    return json.load(f)
    return None


def load_interaction_log(iteration_dir):
    """Load interaction.log from an iteration directory"""
    # Find the timestamp subdirectory
    for subdir in Path(iteration_dir).iterdir():
        if subdir.is_dir():
            log_path = subdir / "interaction.log"
            if log_path.exists():
                with open(log_path, "r") as f:
                    return f.read()
    return None


def render_game_state_conversation(game_state):
    """Render the conversation from game_state.json with proper formatting"""
    if not game_state or "players" not in game_state:
        st.warning("No conversation data available")
        return

    st.markdown('<div class="conversation-container">', unsafe_allow_html=True)

    # Display START marker
    st.markdown('<div class="game-marker">🎮 GAME START</div>', unsafe_allow_html=True)

    # Extract conversations from both players
    player_red = game_state["players"][0]
    player_blue = game_state["players"][1]

    # Interleave conversations
    red_conv = player_red.get("conversation", [])
    blue_conv = player_blue.get("conversation", [])

    # Find assistant messages (player moves)
    red_moves = [msg for msg in red_conv if msg["role"] == "assistant"]
    blue_moves = [msg for msg in blue_conv if msg["role"] == "assistant"]

    # Display moves in order
    total_moves = max(len(red_moves), len(blue_moves))

    for i in range(total_moves):
        if i < len(red_moves):
            render_player_move("Player RED", red_moves[i]["content"], "red")

        if i < len(blue_moves):
            render_player_move("Player BLUE", blue_moves[i]["content"], "blue")

    # Display END marker
    st.markdown('<div class="game-marker">🏁 GAME END</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_player_move(player_name, content, color):
    """Render a single player move with parsed information"""
    # Parse the XML-like tags
    import re

    move_match = re.search(r"<move>(.*?)</move>", content)
    reason_match = re.search(r"<reason>(.*?)</reason>", content, re.DOTALL)
    message_match = re.search(r"<message>(.*?)</message>", content, re.DOTALL)
    answer_match = re.search(r"<player answer>(.*?)</player answer>", content)
    trade_match = re.search(
        r"<newly proposed trade>(.*?)</newly proposed trade>", content
    )

    move = move_match.group(1).strip() if move_match else "Unknown"
    reason = reason_match.group(1).strip() if reason_match else ""
    message = message_match.group(1).strip() if message_match else ""
    answer = answer_match.group(1).strip() if answer_match else "NONE"
    trade = trade_match.group(1).strip() if trade_match else "NONE"

    css_class = f"player-{color}"

    html = f'<div class="{css_class}">'
    html += f"<strong>{player_name}</strong> - Move {move}<br/><br/>"

    if message and message != "":
        html += f"<strong>💬 Message:</strong><br/>{message}<br/><br/>"

    if answer != "NONE":
        if answer == "ACCEPT":
            html += f'<strong>✅ Action:</strong> <span style="color: green; font-weight: bold;">{answer}</span><br/><br/>'
        elif answer == "REJECT":
            html += f'<strong>❌ Action:</strong> <span style="color: red; font-weight: bold;">{answer}</span><br/><br/>'

    if trade != "NONE":
        html += f"<strong>📝 Proposed Trade:</strong><br/>{trade}<br/><br/>"

    if reason and reason != "":
        html += f'<details><summary><strong>💭 View Reasoning</strong></summary><div style="margin-top: 10px; padding: 10px; background-color: rgba(0,0,0,0.05); border-radius: 5px; color: #000;">{reason}</div></details>'

    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)


def render_interaction_log(log_content):
    """Render the raw interaction.log file"""
    st.markdown('<div class="conversation-container">', unsafe_allow_html=True)
    st.code(log_content, language="text")
    st.markdown("</div>", unsafe_allow_html=True)


def render_raw_stats(summary_data, combination_key):
    """Render raw statistics from summary.json"""
    if not summary_data or combination_key not in summary_data:
        st.warning("No statistics available for this combination")
        return

    stats = summary_data[combination_key]

    st.subheader("Raw Summary Data")

    # Display as formatted JSON-like structure
    for key, value in stats.items():
        if isinstance(value, list):
            # For lists, show first few items if long
            if len(value) > 10:
                display_value = (
                    f"[{', '.join(map(str, value[:10]))}... ({len(value)} items)]"
                )
            else:
                display_value = str(value)
        elif isinstance(value, float):
            display_value = f"{value:.4f}"
        else:
            display_value = str(value)

        st.markdown(
            f'<div class="stat-item"><span class="stat-label">{key}:</span><span class="stat-value">{display_value}</span></div>',
            unsafe_allow_html=True,
        )


def main():
    st.title("🎮 Ultimatum Game Visualizer")

    # Initialize session state
    if "selected_combination_idx" not in st.session_state:
        st.session_state.selected_combination_idx = 0
    if "selected_iteration" not in st.session_state:
        st.session_state.selected_iteration = 1

    # Sidebar for navigation
    with st.sidebar:
        st.header("📁 Log Directory")

        # Allow user to input directory path
        default_path = str(
            Path.home()
            / "Projects/LLMs-NegotiationResearch/NegotiationArena-main/.logs/ultimatum_social_behavior_20251225_191511"
        )
        log_dir = st.text_input(
            "Directory Path",
            value=default_path,
            help="Enter the path to the log directory",
        )

        if not Path(log_dir).exists():
            st.error("❌ Directory not found!")
            st.stop()

        st.success("✅ Directory loaded!")

        # Load combinations
        combinations = get_combinations(log_dir)

        if not combinations:
            st.error("❌ No valid combinations found")
            st.stop()

        st.header("🎯 Select Combination")

        # Create scrollable selection boxes
        st.markdown("### Model A")
        model_a_options = sorted(list(set([c["model1"] for c in combinations])))
        selected_model_a = st.selectbox(
            "Model A", model_a_options, label_visibility="collapsed"
        )

        st.markdown("### Model B")
        # Filter model B options based on Model A selection
        model_b_options = sorted(
            list(
                set(
                    [
                        c["model2"]
                        for c in combinations
                        if c["model1"] == selected_model_a
                    ]
                )
            )
        )
        selected_model_b = st.selectbox(
            "Model B", model_b_options, label_visibility="collapsed"
        )

        st.markdown("### Behavior")
        # Filter behavior options based on both model selections
        behavior_options = sorted(
            list(
                set(
                    [
                        c["behavior"]
                        for c in combinations
                        if c["model1"] == selected_model_a
                        and c["model2"] == selected_model_b
                    ]
                )
            )
        )
        selected_behavior = st.selectbox(
            "Behavior", behavior_options, label_visibility="collapsed"
        )

        # Find the matching combination
        selected_combination = None
        for c in combinations:
            if (
                c["model1"] == selected_model_a
                and c["model2"] == selected_model_b
                and c["behavior"] == selected_behavior
            ):
                selected_combination = c
                break

        if not selected_combination:
            st.error("No matching combination found")
            st.stop()

        st.markdown("---")
        st.info(
            f"**Model A:** {selected_combination['model1']}\n\n"
            f"**Model B:** {selected_combination['model2']}\n\n"
            f"**Behavior:** {selected_combination['behavior']}"
        )

    # Main content area
    # Panel 3: Raw Statistics (at the top)
    st.header("📊 Summary Statistics")
    summary_data = load_summary_data(log_dir)
    render_raw_stats(summary_data, selected_combination["key"])

    st.markdown("---")

    # Panel 2: Conversation Viewer
    st.header("💬 Conversation Viewer")

    # Get iterations for selected combination
    iterations = get_iteration_dirs(log_dir, selected_combination["key"])

    if not iterations:
        st.warning("No iterations found for this combination")
        st.stop()

    # Iteration selector with buttons
    st.markdown("### Select Iteration")

    # Create 10 buttons in a row
    cols = st.columns(10)
    for i in range(min(10, len(iterations))):
        with cols[i]:
            if st.button(f"{i + 1}", key=f"iter_btn_{i}", use_container_width=True):
                st.session_state.selected_iteration = i + 1

    # View mode selector
    view_mode = st.radio("View Mode", ["Parsed", "Raw Log"], horizontal=True)

    # Get the selected iteration
    selected_iter_idx = st.session_state.selected_iteration - 1
    if selected_iter_idx >= len(iterations):
        selected_iter_idx = 0
        st.session_state.selected_iteration = 1

    selected_iteration = iterations[selected_iter_idx]

    st.markdown(f"**Showing Iteration {st.session_state.selected_iteration}**")

    # Display conversation based on view mode
    if view_mode == "Parsed":
        game_state = load_game_state(selected_iteration["path"])
        if game_state:
            render_game_state_conversation(game_state)
        else:
            st.error("Could not load game state")
    else:
        log_content = load_interaction_log(selected_iteration["path"])
        if log_content:
            render_interaction_log(log_content)
        else:
            st.error("Could not load interaction log")


if __name__ == "__main__":
    main()
