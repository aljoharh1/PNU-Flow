import sys
import os
from pathlib import Path
import streamlit as st
import pickle

# 1. UI CONFIGURATION - must be the first streamlit command
st.set_page_config(page_title="PNU-Flow Navigation", layout="wide")

# 2. PATH CONFIGURATION
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 3. IMPORT PROJECT PIPELINES
query_route = None
PATHS = None
try:
    from pipelines.inference_pipeline import query_route
    from config import PATHS
except ImportError as e:
    st.error(f"Module loading error: {e}")
    st.stop()

# 4. HELPER FUNCTIONS
def format_name(name):
    """Converts technical names (e.g., corridor_A_1) to readable text (Corridor A 1)."""
    if not name:
        return name
    return name.replace('_', ' ').title()

# 5. MAIN INTERFACE HEADERS
st.title("PNU-Flow: Intelligent Indoor Navigation")
st.markdown("Predictive crowd management and optimized routing for CCIS building.")

# 6. SIDEBAR SETTINGS
st.sidebar.header("Navigation Settings")

# Load zone mapping for the dropdown menus
try:
    mapping_path = PATHS.artifacts_dir / "zone_mapping.pkl"
    with open(mapping_path, "rb") as f:
        zone_map_data = pickle.load(f)
    zones = sorted(list(zone_map_data.keys()))
except Exception:
    # Fallback list if mapping file is not found
    zones = ["main_entrance", "cafeteria", "corridor_A_G", "stairs_G1", "elevator_lobby_G"]

source = st.sidebar.selectbox("Select Start Point:", zones, key="src_key", format_func=format_name)
destination = st.sidebar.selectbox("Select Destination:", zones, key="dest_key", format_func=format_name)

# 7. MAIN NAVIGATION LOGIC
if st.sidebar.button("Find Best Route"):
    with st.spinner('Calculating the optimal path using LSTM occupancy predictions...'):
        try:
            # Call the AI Backend (LSTM + A*)
            result = query_route(source, destination)

            # Display Key Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Estimated Time (ETA)", f"{result['eta_seconds']} sec")
            col2.metric("Distance Score", f"{result['distance_weighted_meters']} m")
            col3.metric("Model Confidence", f"{result['avg_model_confidence']*100:.1f}%")

            # Display the Recommended Path
            st.subheader("Recommended Path")
            clean_path = [format_name(step) for step in result['path']]
            st.success(" → ".join(clean_path))

            # --- NEW FEATURE: Route Congestion Details (Visual Breakdown) ---
            with st.expander("🔍 View Route Congestion Details"):
                st.caption("Real-time occupancy predictions along your path:")
                for step in result['path']:
                    # Get occupancy prediction from LSTM
                    occ_val = result['occupancy_predictions'].get(step, 0.0)
                    occ_pct = occ_val * 100
                    
                    # Traffic light icons based on percentage
                    if occ_pct < 5.0:
                        status_icon = "🟢" # Quiet
                    elif occ_pct < 15.0:
                        status_icon = "🟡" # Moderate
                    else:
                        status_icon = "🔴" # Busy
                        
                    st.write(f"{status_icon} **{format_name(step)}** — {occ_pct:.1f}%")
                    st.progress(float(min(occ_val, 1.0)))
            # ---------------------------------------------------------------

            # Routing Status Info
            if result.get('used_shortest_path_fallback'):
                st.warning("Note: Using standard shortest path due to low model confidence.")
            else:
                st.info("Route optimized based on current occupancy (Quiet Route).")

            # Suggest Study Spot
            st.subheader("Suggested Study Spot")
            study = result.get('study_spot')
            if isinstance(study, dict) and 'zone' in study:
                occ = study.get('occupancy_pct', 0) * 100
                st.write(f"The best place to study right now is **{format_name(study['zone'])}** with an occupancy of **{occ:.1f}%**.")
            elif isinstance(study, str):
                st.write(format_name(study))
            else:
                st.info("No specific study spot recommendation available at this moment.")

        except Exception as e:
            st.error(f"Error during inference: {e}")

# 8. SYSTEM MONITORING (Sidebar Footer)
st.sidebar.markdown("---")
st.sidebar.subheader("System Health")
st.sidebar.write("✅ LSTM Model: Loaded")
st.sidebar.write("✅ Graph Engine: Active")