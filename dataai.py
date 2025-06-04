import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import google.generativeai 
import streamlit.components.v1 as components



common_user_inputs = ["hi", "hello", "hey", "howdy", "greetings", "good morning", "good afternoon", "good evening", "ok", "okay", "bye", "goodbye", "see ya", "thank you", "thanks", "cheers", "how are you", "what's up", "you good", "awesome", "great", "cool", "nice", "perfect", "got it", "understand", "sure", "yes", "no", "help", "support", "info", "tell me more", "please", "what is shree.ai", "tell me about shree.ai", "more about shree.ai", "who is the founder of shree.ai", "who is the ceo of shree.ai", "founder and ceo", "raj singh rajput"]


# Gemini Client Setup
genai = google.generativeai.configure(api_key=st.secrets["GEMINI_API_KEY"])  # Replace with your real Gemini API key
# --- Gemini Interaction ---
def assistantagent(system_prompt, user_prompt):
    prompt = f"{system_prompt}\n\n{user_prompt}"
    time.sleep(5)
    response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
    return response.text.strip().replace("```python", "").replace("```", "").strip()
def chat_agent(message):
    system_prompt = """Introduction
"Hello! Welcome to Data.AI — a virtual data analytics assistant that turns numbers into clear, visual insights.

 I'm data.AI, your intelligent assistant for visual data analytics, proudly powered by shree.ai. I'm designed to help you transform complex data into clear, actionable insights, all for free."

What is shree.ai?
"shree.ai is a multinational company dedicated to empowering businesses like yours. We specialize in creating cutting-edge, custom AI-powered agents meticulously designed to meet your unique operational needs and drive significant business growth."

Who is the Founder & CEO?
"Our innovative vision and strategic direction are led by our esteemed Founder and CEO, Rajsingh Rajput".
"""
    user_prompt = f"""
    {message}
    """
    response = assistantagent(system_prompt, user_prompt)
    return response

# --- Auto-formatting function ---
def auto_format_data(df):
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
                continue
            except (ValueError, TypeError):
                pass
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- Dataset Description ---
def generate_description(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    stats = df.describe(include='all').to_string()
    return f"{info_str}\n\n{stats}"

def safe_execute_and_summarize(insight, code, df):
    try:
        compiled = compile(code, "<string>", "exec")
        local_scope = {"df": df, "np": np}
        exec(compiled, {}, local_scope)
        result = local_scope.get("result", None)

        # Detect empty or meaningless results
        if result is None or (
            isinstance(result, (pd.DataFrame, pd.Series, list, dict)) and len(result) == 0
        ):
            summary = f"There is no meaningful data found for: '{insight}'"
            result = "⚠️ No data to display"
        else:
            summary = representative_agent(insight, result)

    except SyntaxError as se:
        result = f"❌ Syntax Error: {se}"
        summary = "The request could not be processed due to a syntax error."

    except Exception as e:
        result = f"❌ Runtime Error: {e}"
        summary = "Something went wrong while processing the insight."

    return result, summary


# --- Coder Agent ---
def coder_agent(insight, description, data_sample):
    system_prompt = """
You are a professional Python data analyst assistant. Your task is to write clean, correct, and efficient Pandas code to fulfill a specific data insight request.
"""
    user_prompt = f"""
🎯 OBJECTIVE:
Write a Python code snippet using Pandas to extract the following data insight from a DataFrame named `df`.

📌 Insight:
{insight}

📘 CONTEXT:
You have access to a dataset (already loaded into the DataFrame `df`). A short description and sample of this dataset is provided.

📄 DATASET DESCRIPTION:
{description}

🧪 SAMPLE DATA:
{data_sample}

⚠️ RULES:
- DO NOT create or redefine the DataFrame `df`.
- DO NOT include mock data or placeholders.
- DO NOT use any visualization or plotting libraries.
- DO NOT include comments or explanations.
- ONLY use the `df` DataFrame for analysis using Pandas (NumPy is allowed if needed).
- Handle missing values if relevant to the logic.
- Assign the final result to a variable named `result`.

✅ OUTPUT FORMAT:
Return a single block of valid, executable Python code only, and ensure the final result is stored in a variable called `result`.
"""
    code = assistantagent(system_prompt, user_prompt)
    code = code.strip().replace("```python", "").replace("```", "").replace("python", "").strip()
    return code

# --- Review Agent ---
def review_agent(insight, code):
    system_prompt = "You are an expert Python code reviewer. Your task is to evaluate a Pandas code snippet written to extract a specific data insight."
    user_prompt = f"""
🎯 Insight to Evaluate:
{insight}

💻 Code to Review:
{code}

✅ REVIEW GUIDELINES:
- Check if the code fulfills the insight accurately.
- Identify any issues in logic, syntax, error handling, or performance.
- Tag each issue clearly as [minor], [major], or [critical].
- Ignore lack of visualization or code comments.
- If the code is flawless, return exactly: APPROVED
"""
    code = assistantagent(system_prompt, user_prompt)
    code = code.strip().replace("python", "").replace("", "").strip()
    return code

# --- Code Editor Agent ---
def code_editor_agent(insight, description, data_sample, original_code, issues):
    system_prompt = """
You are a highly skilled Python code editor. Your job is to fix issues in a Pandas code snippet so that it correctly extracts a requested insight without introducing new errors.
"""
    user_prompt = f"""
🎯 GOAL:
Fix and improve the following Pandas code to extract this insight:
{insight}

📄 DATASET DESCRIPTION:
{description}

🧪 SAMPLE DATA:
{data_sample}

💻 ORIGINAL CODE:
{original_code}

🚩 ISSUES FOUND:
{issues}

⚙️ RULES:
- ignore minor review issues.
- DO NOT create or redefine the DataFrame `df`.
- DO NOT include mock data or placeholder values.
- ONLY use Pandas (and NumPy if necessary).
- DO NOT use visualization or plotting libraries.
- DO NOT add comments or explanations.
- Fix all listed issues with accurate and efficient logic.
- Add error handling if relevant.
- Always handle or drop missing values where applicable.
- Ensure the final result is stored in a variable called `result`.

✅ OUTPUT FORMAT:
Return only the final corrected Python code block — clean and executable.
"""
    return assistantagent(system_prompt, user_prompt)

# --- Representative Agent ---
def representative_agent(insight, code_output):
    system_prompt = """
You are an elite business data analyst and executive insights communicator.

Your role is to clearly and professionally summarize data analysis results for business stakeholders, using natural language — no code or jargon.

You will be given:
- A business question or data insight to extract.
- The final result or output from a Python data analysis.

Your job:
✅ Understand what the output reveals in the context of the insight.
✅ Provide a concise, clear, and impressive summary of the result.
✅ Communicate it like a consultant, analyst, or insights specialist would in a presentation or executive report.

If the result is empty or None, state that no meaningful data was found.

Respond only with the final summary — no explanations or code.
"""
    user_prompt = f"""
📌 Business Insight Requested:
{insight}

📈 Final Result from Code Execution:
{code_output}
"""
    return assistantagent(system_prompt, user_prompt)

# --- Manager Agent ---
def manager_agent(insight, description, data_sample):
    max_retries = 8
    code = coder_agent(insight, description, data_sample)

    for attempt in range(max_retries):
        review = review_agent(insight, code)
        if "APPROVED" in review:
            return code, review
        code = code_editor_agent(insight, description, data_sample, code, review)

    return code, review

# --- Streamlit App ---
st.set_page_config("AI Insight Generator")
def set_page_title(title):
    col1, col2, col3 = st.columns([1, 6, 1]) # Adjust the ratio as needed
    with col2:
        st.markdown(f"<h1 style='text-align: center;'>{title}</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 4, 1]) # Adjust the ratio as needed
    with col2:
        st.markdown(f"<h3 style='text-align: center;'>A Virtual Data Analytics</h3>", unsafe_allow_html=True)

set_page_title("Data.AI")



uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Auto formatting if needed
    df = auto_format_data(df)

    st.subheader("Preview of Data")
    st.dataframe(df.head())

    st.subheader("Cleaned Data")
    st.dataframe(df)

    # Download cleaned data as CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
    label="📥 Download",
    data=csv,
    file_name="cleaned_data.csv",
    mime="text/csv",
    help="Click to download the cleaned dataset."
)

    description = generate_description(df)[:2500]
    data_sample = df.head().to_string()

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    # Display chat history
    with st.container():
        for chat in st.session_state.chat_history:
            with st.chat_message(chat["role"]):
                st.markdown(chat["content"])

    # Chat input
    user_input = st.chat_input("Ask a data question...")
    if user_input:
        # Save user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(f"<div style='background:none;padding:0;margin:0'>{user_input}</div>", unsafe_allow_html=True)

        # Friendly greetings
        greetings = common_user_inputs
        if any(greet in user_input.lower() for greet in greetings):
            ai_reply = chat_agent(user_input)
            with st.chat_message("assistant"):
                st.markdown(ai_reply)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ai_reply
            })
            st.stop()

        # Clarification for vague questions
        if st.session_state.pending_question:
            full_insight = f"{st.session_state.pending_question}, clarified as: {user_input}"
            st.session_state.pending_question = None
        else:
            vague_keywords = ["top", "most", "best", "worst", "high", "low", "category", "segment"]
            if any(word in user_input.lower() for word in vague_keywords) and "by" not in user_input.lower():
                clarify_text = "🤖 Could you clarify — do you mean top by *sales*, *orders*, *profit*, or something else?"
                with st.chat_message("assistant"):
                    st.markdown(clarify_text)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": clarify_text
                })
                st.session_state.pending_question = user_input
                st.stop()
            else:
                full_insight = user_input

        # Proceed with Insight Generation
        with st.spinner("Generating insight..."):
            try:
                code, review = manager_agent(full_insight, description, data_sample)

                if "APPROVED" in review:
                    compiled = compile(code, "<string>", "exec")
                    local_scope = {"df": df, "np": np}
                    exec(compiled, {}, local_scope)
                    result = local_scope.get("result", None)
                    summary = representative_agent(full_insight, result)

                    if result is None or (isinstance(result, pd.DataFrame) and result.empty):
                        result = "❗ No data found for this query."
                        summary = "There is no meaningful data to show for this insight."

                else:
                    result, summary = safe_execute_and_summarize(full_insight, code, df)

            except SyntaxError as se:
                result = "⚠️ Could not process this request due to a code syntax issue."
                summary = "The system couldn't process your request due to a technical issue."

            except Exception as e:
                result = "⚠️ An unexpected error occurred while processing the request."
                summary = "Something went wrong internally. Please try again with a different insight."

        # Display AI response
        with st.chat_message("assistant"):
            st.markdown(f"<div style='background:none;padding:0;margin:0'>{result}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background:none;padding:0;margin:0;color:green;font-weight:600'>{summary}</div>", unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"{result}\n\n✅ {summary}"
        })

    # Auto-scroll to bottom
    components.html("""
        <script>
            const chatContainer = window.parent.document.querySelector('.main');
            if (chatContainer) {
                chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
            }
        </script>
    """, height=0)
