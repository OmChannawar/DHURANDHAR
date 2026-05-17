import streamlit as st
import time
import pandas as pd
from ml_backend import predict_all_models

# --- Page Config ---
st.set_page_config(
    page_title="DHURANDHAR | Live AI Analysis",
    page_icon="📰",
    layout="wide",  # Wide layout for side-by-side dashboard capability
    initial_sidebar_state="collapsed"
)

# --- Custom CSS Theme ---
st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #1E293B;
    }
    
    /* Input Text Area Customization */
    .stTextArea textarea {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
        padding: 1rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 1px #3B82F6 !important;
    }
    </style>
""", unsafe_allow_html=True)


# --- UI Helper Functions ---
def generate_table_html(results):
    """Generates the modern live-updating Prediction Table with integrated progress bars."""
    html = """
    <table style="width:100%; border-collapse: collapse; text-align: left; font-family: 'Inter', sans-serif; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); border-radius: 8px; overflow: hidden;">
        <tr style="background-color: #F8FAFC; border-bottom: 1px solid #E2E8F0;">
            <th style="padding: 14px 16px; color: #475569; font-weight: 600; font-size: 0.9rem; text-transform: uppercase;">Model Engine</th>
            <th style="padding: 14px 16px; color: #475569; font-weight: 600; font-size: 0.9rem; text-transform: uppercase;">Prediction</th>
            <th style="padding: 14px 16px; color: #475569; font-weight: 600; font-size: 0.9rem; text-transform: uppercase;">Probability</th>
        </tr>
    """
    for idx, r in enumerate(results):
        bg_color = "#FFFFFF" if idx % 2 == 0 else "#F8FAFC"
        color = "#16A34A" if r["label"] == "Real" else "#DC2626"
        html += f"""
        <tr style="background-color: {bg_color}; border-bottom: 1px solid #F1F5F9;">
            <td style="padding: 12px 16px; color: #1E293B; font-weight: 500;">{r['model']}</td>
            <td style="padding: 12px 16px; font-weight: 800; color: {color};">{r['label']}</td>
            <td style="padding: 12px 16px; color: #334155;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="flex-grow: 1; background: #E2E8F0; border-radius: 4px; height: 6px; overflow: hidden;">
                        <div style="width: {r['probability']}%; background: {color}; height: 100%;"></div>
                    </div>
                    <span style="min-width: 45px; font-size: 0.9rem; font-weight: 500;">{r['probability']:.1f}%</span>
                </div>
            </td>
        </tr>
        """
    html += "</table>"
    return html


def generate_metrics_html(avg_prob, final_label):
    """Generates the styled KPI cards for Average Probability and the Final Decision."""
    if final_label == "Real":
        color = "#16A34A"
        bg = "#DCFCE7"
        border = "#bbf7d0"
    elif final_label == "Fake":
        color = "#DC2626"
        bg = "#FEE2E2"
        border = "#fecaca"
    else:
        color = "#475569"
        bg = "#F1F5F9"
        border = "#e2e8f0"

    return f"""
    <div style="display: flex; gap: 20px; margin-top: 24px;">
        <div style="flex: 1; padding: 24px; background: {bg}; border-radius: 12px; border: 1px solid {border}; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
            <div style="color: {color}; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Final Ensemble Decision</div>
            <div style="color: {color}; font-size: 2.2rem; font-weight: 800; margin-top: 8px;">{final_label.upper()}</div>
        </div>
        <div style="flex: 1; padding: 24px; background: #F8FAFC; border-radius: 12px; border: 1px solid #E2E8F0; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
            <div style="color: #64748B; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Average Confidence</div>
            <div style="color: #0F172A; font-size: 2.2rem; font-weight: 800; margin-top: 8px;">{avg_prob:.1f}%</div>
        </div>
    </div>
    """


def main():
    # --- HERO HEADER (DHURANDHAR Branding) ---
    st.markdown("""
    <div style="text-align: center; margin-top: 1rem; margin-bottom: 3.5rem;">
        <h1 style="
            font-size: 4rem; 
            font-weight: 900; 
            background: linear-gradient(135deg, #2563EB 0%, #06B6D4 100%); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
            letter-spacing: 0.12em;
            margin-bottom: 0;
            text-shadow: 0px 4px 15px rgba(37, 99, 235, 0.1);">
            DHURANDHAR
        </h1>
        <h3 style="color: #64748B; font-style: italic; font-weight: 400; margin-top: 0.5rem; margin-bottom: 0.5rem; font-size: 1.2rem;">
            "Ghaflat ki khabar... ya khabar ki ghaflat?"
        </h3>
        <p style="color: #94A3B8; font-size: 1rem; letter-spacing: 0.1em; font-weight: 600; text-transform: uppercase;">
            AI-Powered Fake News Detection System
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- SIDE-BY-SIDE DASHBOARD LAYOUT ---
    col1, col2 = st.columns([1, 1.8], gap="large")

    with col1:
        st.markdown("<h3 style='color: #1E293B; font-weight: 700; font-size: 1.3rem; margin-bottom: 0.5rem;'>📝 Content Source</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748B; font-size: 0.95rem; margin-bottom: 1rem;'>Paste a news article snippet or headline below. The system will dispatch it to all available ML agents simultaneously.</p>", unsafe_allow_html=True)
        
        news_text = st.text_area(
            "News Content:", 
            height=280, 
            placeholder="e.g., 'A shocking new report confirms that staring at screens gives you superpowers...'",
            label_visibility="collapsed"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 Run Live Analytics", type="primary", use_container_width=True)

    with col2:
        st.markdown("<h3 style='color: #1E293B; font-weight: 700; font-size: 1.3rem; margin-bottom: 1.5rem;'>📊 Live Agent Analytics</h3>", unsafe_allow_html=True)
        
        # Placeholders for dynamic content updating
        progress_bar = st.empty()
        status_text = st.empty()
        table_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        # Initial empty state card
        if not analyze_btn:
            table_placeholder.markdown("""
            <div style="padding: 40px; text-align: center; border: 2px dashed #E2E8F0; border-radius: 12px; background: #F8FAFC;">
                <h4 style="margin: 0; color: #64748B; font-weight: 600;">System Awaiting Input</h4>
                <p style="margin-top: 8px; font-size: 0.95rem; color: #94A3B8;">Enter content and launch analytics to view the live multi-model evaluation dashboard.</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            if not news_text.strip():
                st.error("Text payload empty. Please enter content to proceed.")
            elif len(news_text.strip()) < 10:
                st.warning("Payload length insufficient for reliable analysis. Add more text.")
            else:
                with st.spinner("Dispatching payload to ML engines..."):
                    pb = progress_bar.progress(0)
                    status_text.markdown("<p style='color: #2563EB; font-weight: 600; font-size: 0.95rem;'>Processing real-time evaluation...</p>", unsafe_allow_html=True)
                    
                    total_expected_models = 9
                    
                    # DYNAMIC ANIMATION: Loop through the generator as models yield results
                    for i, data in enumerate(predict_all_models(news_text)):
                        
                        # 1. Increment progress
                        current_count = i + 1
                        progress_pct = min(100, int((current_count / total_expected_models) * 100))
                        pb.progress(progress_pct)
                        
                        # 2. Update Table Live
                        table_html = generate_table_html(data["results"])
                        table_placeholder.markdown(table_html, unsafe_allow_html=True)
                        
                        # 3. Update Metrics Live
                        metrics_html = generate_metrics_html(data["average_probability"], data["final_label"])
                        metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)
                        
                        # Small UI delay to make the sequential cascade feel premium and readable
                        time.sleep(0.4)
                        
                    # Clean up statuses upon finish
                    status_text.empty()
                    progress_bar.empty()
                    st.success("Analysis complete. Multi-model consensus reached.")

if __name__ == "__main__":
    main()