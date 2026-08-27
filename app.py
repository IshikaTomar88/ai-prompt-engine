import streamlit as st
import time
import pandas as pd
import plotly.express as px
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise AI Prompt Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI design
st.markdown("""
    <style>
    /* Main Layout */
    .stApp { max-width: 1300px; margin: 0 auto; }
    
    /* Custom Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Code/JSON container overrides */
    .stJson { background-color: #0f172a !important; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. PYDANTIC SCHEMAS (Structured API Outputs)
# ------------------------------------------------------------------
class CustomerInsight(BaseModel):
    sentiment: str = Field(description="Overall sentiment: Positive, Neutral, or Negative")
    urgency_score: int = Field(description="Urgency score from 1 (low) to 10 (critical)")
    core_intent: str = Field(description="Primary user intent or root problem summary")
    actionable_steps: list[str] = Field(description="3 strategic actions to resolve the issue")
    optimized_reply: str = Field(description="Polished email response tailored to user context")

# ------------------------------------------------------------------
# 3. SIDEBAR CONFIGURATION
# ------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Engine Settings")

# Safe API key retrieval
api_key_from_secrets = st.secrets.get("OPENAI_API_KEY", "")
user_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    value=api_key_from_secrets if not api_key_from_secrets.endswith("here") else "",
    placeholder="sk-proj-...",
    help="Enter your valid OpenAI API Key (sk-...)"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Model Parameters")

selected_model = st.sidebar.selectbox(
    "Model Core",
    ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
)

temperature = st.sidebar.slider("Temperature (Creativity)", 0.0, 1.0, 0.2, 0.05)
max_tokens = st.sidebar.slider("Max Response Tokens", 250, 2000, 800, 50)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Using `gpt-4o-mini` offers the fastest latency and lowest costs for structured extraction.")

# ------------------------------------------------------------------
# 4. MAIN DASHBOARD HEADER
# ------------------------------------------------------------------
st.title("⚡ Enterprise Prompt Studio & API Engine")
st.caption("Production-ready LangChain pipeline featuring Pydantic JSON validation, latency diagnostics, and execution analytics.")

st.markdown("---")

col_in, col_out = st.columns([1, 1.1], gap="large")

# ------------------------------------------------------------------
# 5. INPUT PIPELINE PANEL
# ------------------------------------------------------------------
with col_in:
    st.subheader("📥 Input Pipeline Configuration")
    
    target_role = st.selectbox(
        "System Persona",
        [
            "Senior Customer Support Lead",
            "Technical Product Manager",
            "Executive Copywriter",
            "Security & Compliance Officer"
        ]
    )
    
    tone = st.select_slider(
        "Response Tone",
        options=["Strictly Professional", "Empathetic & Warm", "Direct & Concise", "Technical & Analytical"]
    )

    raw_text = st.text_area(
        "Raw Unstructured Text",
        height=200,
        placeholder="Paste customer emails, tickets, or unstructured inputs...",
        value="Our cloud servers crashed during peak hours today! We lost almost $4,000 in sales. I need an immediate refund for our monthly plan and a call with your tech leads ASAP. Order ID: #99482."
    )

    run_btn = st.button("🚀 Process API Pipeline", type="primary", use_container_width=True)

# ------------------------------------------------------------------
# 6. PIPELINE EXECUTION & ANALYTICS
# ------------------------------------------------------------------
with col_out:
    st.subheader("📤 Structured Output & Diagnostics")
    
    if run_btn:
        # Validate API Key first before calling OpenAI
        if not user_api_key or "here" in user_api_key:
            st.error("🔑 **Invalid or Missing API Key!**")
            st.warning("Please enter a valid OpenAI API key (`sk-proj-...`) in the sidebar settings or set it in your Streamlit secrets.")
        elif not raw_text.strip():
            st.warning("⚠️ Input text cannot be empty.")
        else:
            with st.spinner("Executing LangChain chain & validating Pydantic schemas..."):
                start_time = time.time()
                
                try:
                    # Initialize Model Core
                    llm = ChatOpenAI(
                        model=selected_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=user_api_key
                    )
                    
                    # Construct Structured Prompt Chain
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are acting as a {role}. Process input with high precision and output strictly according to the schema."),
                        ("user", "Requested Tone: {tone}\n\nInput Content:\n{input}")
                    ])

                    structured_llm = llm.with_structured_output(CustomerInsight)
                    chain = prompt | structured_llm

                    # Execute
                    response: CustomerInsight = chain.invoke({
                        "role": target_role,
                        "tone": tone,
                        "input": raw_text
                    })
                    
                    latency = round(time.time() - start_time, 2)
                    char_count = len(raw_text)
                    est_tokens = round(char_count / 4)

                    # --- ADVANCED UI METRICS CARDS ---
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{response.sentiment}</div><div class="metric-label">Sentiment</div></div>', unsafe_allow_html=True)
                    with m2:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{response.urgency_score}/10</div><div class="metric-label">Urgency</div></div>', unsafe_allow_html=True)
                    with m3:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{latency}s</div><div class="metric-label">Latency</div></div>', unsafe_allow_html=True)
                    with m4:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">~{est_tokens}</div><div class="metric-label">Tokens</div></div>', unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # --- TABBED RESULTS PRESENTATION ---
                    tab_reply, tab_action, tab_analytics, tab_json = st.tabs([
                        "✉️ Drafted Reply",
                        "📋 Action Items",
                        "📊 Analytics",
                        "🔍 Raw JSON Schema"
                    ])

                    with tab_reply:
                        st.markdown("**Identified Intent:**")
                        st.info(response.core_intent)
                        st.markdown("**Generated Response:**")
                        st.success(response.optimized_reply)

                    with tab_action:
                        st.markdown("**Recommended Resolution Plan:**")
                        for idx, step in enumerate(response.actionable_steps, 1):
                            st.markdown(f"**{idx}.** {step}")

                    with tab_analytics:
                        st.markdown("**Execution Diagnostics**")
                        
                        # Plotly visual chart for latency breakdown
                        metrics_df = pd.DataFrame({
                            "Metric": ["Est. Input Tokens", "Execution Speed (ms)", "Urgency Level (x10)"],
                            "Value": [est_tokens, latency * 1000, response.urgency_score * 10]
                        })
                        fig = px.bar(
                            metrics_df,
                            x="Metric",
                            y="Value",
                            color="Metric",
                            title="Pipeline Execution Overview",
                            color_discrete_sequence=["#38bdf8", "#818cf8", "#f43f5e"]
                        )
                        fig.update_layout(showlegend=False, height=300, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)

                    with tab_json:
                        st.markdown("**JSON Output Structure:**")
                        st.json(response.model_dump())
                        
                        # Direct JSON Download button
                        st.download_button(
                            label="📥 Download Output JSON",
                            data=json.dumps(response.model_dump(), indent=2),
                            file_name="ai_pipeline_output.json",
                            mime="application/json"
                        )

                except Exception as e:
                    err_msg = str(e)
                    if "401" in err_msg or "invalid_api_key" in err_msg:
                        st.error("❌ **Authentication Error (401)**")
                        st.info("Your OpenAI API Key is incorrect or expired. Check your key at https://platform.openai.com/account/api-keys")
                    else:
                        st.error(f"Execution Error: {err_msg}")
    else:
        st.info("👈 Set your parameters on the left and click **Process API Pipeline** to view output.")
