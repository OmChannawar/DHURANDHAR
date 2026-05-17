import streamlit as st
import time
from ml_backend import predict_all_models

# --- Page Config ---
st.set_page_config(
    page_title="DHURANDHAR | Live AI Analysis",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS Theme ---
st.markdown("""
    <style>
    /* Force pure white background and remove Streamlit's dark mode defaults */
    [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF !important;
    }
    [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
    }
    .stApp {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }
    
    /* Input Text Area Customization */
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        border: 2px solid #E2E8F0 !important;
        border-radius: 12px !important;
        color: #0F172A !important;
        padding: 1.5rem !important;
        font-size: 1.05rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1) !important;
    }

    /* Custom Predict Button */
    div.stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #8B5CF6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.05em !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
    }
    </style>
""", unsafe_allow_html=True)


# --- UI Helper Functions ---
def generate_meter_html(results):
    """Generates the visual scoreboard with gauge/speedometer charts."""
    html = '<div style="display: flex; flex-direction: column; gap: 1rem;">'
    for r in results:
        prob = r['probability']
        label = r['label']
        model = r['model']
        
        # Color logic
        if label == "Real":
            color = "#16A34A" # Green for real
        elif label == "Fake":
            color = "#DC2626" # Red for fake
        else:
            color = "#64748B"
            
        # Override for moderate confidence
        if 40 <= prob < 75:
            color = "#F59E0B" # Orange
            
        dash_array = 125.6
        dash_offset = dash_array - (dash_array * prob / 100)
        
        html += f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h4 style="margin: 0; color: #1E293B; font-size: 1.1rem; font-weight: 700;">{model}</h4>
                <p style="margin: 4px 0 0 0; color: {color}; font-weight: 800; font-size: 0.95rem; text-transform: uppercase;">{label}</p>
            </div>
            <div style="width: 120px; text-align: center;">
                <svg viewBox="0 0 100 55" style="width: 100%; display: block; margin: 0 auto; overflow: visible;">
                    <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="#F1F5F9" stroke-width="8" stroke-linecap="round"></path>
                    <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="{color}" stroke-width="8" stroke-linecap="round" stroke-dasharray="{dash_array}" stroke-dashoffset="{dash_offset}" style="transition: stroke-dashoffset 1s ease-out;"></path>
                    <text x="50" y="45" font-family="'Inter', sans-serif" font-size="16" font-weight="800" fill="#1E293B" text-anchor="middle">{prob:.1f}%</text>
                </svg>
            </div>
        </div>
        """
    html += '</div>'
    return html


def generate_metrics_html(avg_prob, final_label):
    """Generates the styled bottom score summary."""
    if final_label == "Real":
        color = "#16A34A"
        bg = "#F0FDF4"
        border = "#BBF7D0"
    elif final_label == "Fake":
        color = "#DC2626"
        bg = "#FEF2F2"
        border = "#FECACA"
    else:
        color = "#475569"
        bg = "#F8FAFC"
        border = "#E2E8F0"

    return f"""
    <div style="display: flex; gap: 20px; margin-top: 1.5rem;">
        <div style="flex: 1; padding: 20px; background: #FFFFFF; border-radius: 12px; border: 1px solid #E2E8F0; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
            <div style="color: #64748B; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Average Confidence</div>
            <div style="color: #1E293B; font-size: 2rem; font-weight: 800; margin-top: 4px;">{avg_prob:.1f}%</div>
        </div>
        <div style="flex: 1; padding: 20px; background: {bg}; border-radius: 12px; border: 1px solid {border}; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
            <div style="color: {color}; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Final Prediction</div>
            <div style="color: {color}; font-size: 2rem; font-weight: 900; margin-top: 4px;">{final_label.upper()} NEWS</div>
        </div>
    </div>
    """


def main():
    if "prediction_results" not in st.session_state:
        st.session_state.prediction_results = None

    # --- 1. TITLE SECTION (HERO HEADER) ---
    st.markdown("""
    <div style="text-align: center; margin-top: 1rem; margin-bottom: 4rem;">
        <h1 style="
            font-size: 4.5rem; 
            font-weight: 900; 
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #1E3A8A 0%, #7C3AED 100%); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
            letter-spacing: 0.15em;
            margin-bottom: 0;
            text-shadow: 0px 4px 25px rgba(124, 58, 237, 0.2);">
            DHURANDHAR
        </h1>
        <h3 style="color: #64748B; font-style: italic; font-weight: 400; margin-top: 0.5rem; margin-bottom: 0.5rem; font-size: 1.4rem;">
            "Ghaflat ki khabar... ya khabar ki ghaflat?"
        </h3>
        <div style="display: flex; justify-content: center; align-items: center; margin: 1rem 0;">
            <div style="height: 1px; width: 50px; background: #CBD5E1;"></div>
            <div style="margin: 0 15px; width: 8px; height: 8px; transform: rotate(45deg); background: #3B82F6;"></div>
            <div style="height: 1px; width: 50px; background: #CBD5E1;"></div>
        </div>
        <p style="color: #475569; font-size: 1.1rem; letter-spacing: 0.15em; font-weight: 600; text-transform: uppercase;">
            AI-Powered Fake News Detection System
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- SIDE-BY-SIDE DASHBOARD LAYOUT ---
    col1, col2 = st.columns([1, 1.3], gap="large")

    # --- 2. LEFT PANEL (INPUT AREA) ---
    with col1:
        news_text = st.text_area(
            "News Content:", 
            height=300, 
            placeholder="Paste or type your news article here...",
            label_visibility="collapsed"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("⚡ PREDICT", use_container_width=True)
        st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 0.85rem; margin-top: 0.5rem;'>🔒 Your text is used only for prediction.</p>", unsafe_allow_html=True)

    # --- 3. RIGHT PANEL (MODEL SCOREBOARD) ---
    with col2:
        # Placeholders for dynamic content updating
        progress_bar = st.empty()
        status_text = st.empty()
        scoreboard_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        # Action triggered by Predict button
        if analyze_btn:
            if not news_text.strip():
                st.error("Text payload empty. Please enter content to proceed.")
            else:
                st.session_state.prediction_results = None # Clear previous
                with st.spinner("Analyzing article..."):
                    pb = progress_bar.progress(0)
                    status_text.markdown("<p style='color: #3B82F6; font-weight: 600; font-size: 0.95rem; text-align: center;'>Activating AI Engines...</p>", unsafe_allow_html=True)
                    
                    total_expected_models = 5 # KNN, SVM, AdaBoost, XGBoost, Gradient Boosting
                    
                    # DYNAMIC ANIMATION: Loop through the generator as models yield results
                    for i, data in enumerate(predict_all_models(news_text)):
                        
                        # Increment progress
                        current_count = i + 1
                        progress_pct = min(100, int((current_count / total_expected_models) * 100))
                        pb.progress(progress_pct)
                        
                        # Save state
                        st.session_state.prediction_results = data
                        
                        # Update Scoreboard Live
                        meter_html = generate_meter_html(st.session_state.prediction_results["results"])
                        scoreboard_placeholder.markdown(meter_html, unsafe_allow_html=True)
                        
                        # Update Metrics Live
                        metrics_html = generate_metrics_html(st.session_state.prediction_results["average_probability"], st.session_state.prediction_results["final_label"])
                        metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)
                        
                        # Smooth animation delay
                        time.sleep(0.5)
                        
                    # Clean up statuses upon finish
                    status_text.empty()
                    progress_bar.empty()

        # Render persisted results if not actively predicting
        if st.session_state.prediction_results is not None and not analyze_btn:
            data = st.session_state.prediction_results
            meter_html = generate_meter_html(data["results"])
            scoreboard_placeholder.markdown(meter_html, unsafe_allow_html=True)
            
            metrics_html = generate_metrics_html(data["average_probability"], data["final_label"])
            metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)
            
        elif st.session_state.prediction_results is None and not analyze_btn:
            # Initial empty state card
            scoreboard_placeholder.markdown("""
            <div style="padding: 60px 40px; text-align: center; border: 1px solid #E2E8F0; border-radius: 16px; background: #FFFFFF; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
                <div style="width: 48px; height: 48px; margin: 0 auto 16px auto; background: #F8FAFC; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 24px;">📊</span>
                </div>
                <h4 style="margin: 0; color: #1E293B; font-weight: 700; font-size: 1.2rem;">Model Scoreboard</h4>
                <p style="margin-top: 8px; font-size: 0.95rem; color: #64748B;">Enter text and click predict to see live AI analysis.</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()