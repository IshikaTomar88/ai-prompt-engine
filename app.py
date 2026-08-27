import streamlit as st
import time
import pandas as pd
import plotly.express as px
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Streamlit Page Setup
st.set_page_config(
    page_title="Enterprise AI Prompt Engine",
    page_icon="⚡",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .stApp { max-width: 1250px; margin: 0 auto; }
    </style>
""", unsafe_allow_html=True)

# Pydantic Schema Definition
class CustomerInsight(BaseModel):
    sentiment: str = Field(description="Overall sentiment: Positive, Neutral, or Negative")
    urgency_score: int = Field(description="Urgency score from 1 (low) to 10 (critical)")
    core_intent: str = Field(description="Primary user intent or root problem")
    actionable_steps: list[str] = Field(description="3 strategic actions to resolve the issue")
    optimized_reply: str = Field(description="Polished email response tailored to user context")

# Sidebar Setup
st.sidebar.title("⚙️ Engine Configuration")

# Extract API key safely from secrets or user input
api_key_from_secrets = st.secrets.get("OPENAI_API_KEY", "")
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    value=api_key_from_secrets,
    help="Provided via .streamlit/secrets.toml or input here manually."
)

selected_model = st.sidebar.selectbox(
    "LLM Model Core",
    ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
)

temperature = st.sidebar.slider("Temperature (Creativity)", 0.0, 1.0, 0.2, 0.05)

# Main Dashboard Interface
st.title("⚡ Enterprise Prompt Engineering & API Pipeline")
st.caption("Production-grade LangChain pipeline featuring structured JSON outputs, latency tracking, and analytical execution.")

st.markdown("---")

col_in, col_out = st.columns([1, 1], gap="medium")

with col_in:
    st.subheader("📥 Input Pipeline Configuration")
    
    target_role = st.selectbox(
        "System Persona",
        ["Senior Customer Support Lead", "Technical Product Manager", "Executive Copywriter"]
    )
    
    tone = st.select_slider(
        "Response Tone",
        options=["Strictly Professional", "Empathetic & Warm", "Direct & Concise"]
    )

    raw_text = st.text_area(
        "Raw Unstructured Text",
        height=180,
        placeholder="Paste customer emails, bug reports, or user requests here...",
        value="Our platform went down during peak sales hours today! I need an immediate refund for our enterprise subscription and a direct meeting with your tech leads. Order ID: #88412."
    )

    run_btn = st.button("🚀 Process API Pipeline", type="primary", use_container_width=True)

# Pipeline Execution
if run_btn:
    if not openai_api_key:
        st.error("🔑 API Key Required! Please add your OpenAI key to sidebar or secrets.")
    elif not raw_text.strip():
        st.warning("⚠️ Input text cannot be empty.")
    else:
        with col_out:
            st.subheader("📤 Structured Pipeline Analytics")
            
            with st.spinner("Executing chain & validating Pydantic schemas..."):
                start_time = time.time()
                
                try:
                    llm = ChatOpenAI(
                        model=selected_model,
                        temperature=temperature,
                        api_key=openai_api_key
                    )
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are acting as a {role}. Analyze the user's input with strict precision."),
                        ("user", "Tone requested: {tone}\n\nInput Content:\n{input}")
                    ])

                    structured_llm = llm.with_structured_output(CustomerInsight)
                    chain = prompt | structured_llm

                    response: CustomerInsight = chain.invoke({
                        "role": target_role,
                        "tone": tone,
                        "input": raw_text
                    })
                    
                    latency = round(time.time() - start_time, 2)

                    # Display Metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Sentiment", response.sentiment)
                    m2.metric("Urgency", f"{response.urgency_score}/10")
                    m3.metric("Latency", f"{latency}s")

                    # Tabbed Data Presentation
                    tab_parsed, tab_action, tab_json = st.tabs(["📝 Generated Output", "📋 Action Plan", "🔍 Raw JSON"])

                    with tab_parsed:
                        st.markdown("**Core Intent Identified:**")
                        st.info(response.core_intent)
                        
                        st.markdown("**Generated Response:**")
                        st.success(response.optimized_reply)

                    with tab_action:
                        st.markdown("**Recommended Resolution Steps:**")
                        for idx, step in enumerate(response.actionable_steps, 1):
                            st.write(f"**{idx}.** {step}")

                    with tab_json:
                        st.json(response.model_dump())

                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")
