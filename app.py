# app.py
import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"  # keep vision deps out of the import path

import io
import time
import pandas as pd
import streamlit as st

# LangChain bits (agent is created in the UI from your existing LLM)
from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# ---- try to import your existing LLM wrapper (no code changes needed) ----
hf_llm = None
import_error = None
try:
    # expects database_agent.py to define `hf_llm` at module level (as in your notebook/py)
    from database_agent import hf_llm as _hf_llm
    hf_llm = _hf_llm
except Exception as e:
    import traceback
    import_error = traceback.format_exc()

@st.cache_resource(show_spinner=False)
def get_llm():
    if hf_llm is None:
        raise RuntimeError(
            "Could not import `hf_llm` from database_agent.py.\n\n"
            + (import_error or "No traceback available.")
        )
    return hf_llm

@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes, sep: str, encoding: str, preview_rows: int = 300):
    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, encoding=encoding)
    # light normalization for friendlier queries
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df, df.head(min(preview_rows, len(df)))

st.set_page_config(page_title="Chat with your CSV", page_icon="📊", layout="wide")
st.title("📊 Chat with your CSV (LangChain)")

with st.sidebar:
    st.header("Agent settings")
    st.caption("Uses your existing `hf_llm` and builds a Pandas agent at runtime.")
    allow_dangerous = st.checkbox(
        "Allow Python execution (required by Pandas agent)",
        value=True,
        help="The Pandas DataFrame agent uses a Python REPL tool under the hood."
    )
    max_new_tokens = st.slider("Max new tokens (answer length)", 32, 1024, 256, 32)
    max_iterations = st.slider("Max agent iterations", 1, 12, 6, 1)
    max_exec_time = st.slider("Max execution time (sec)", 5, 120, 60, 5)
    st.divider()
    st.markdown("**LLM source:** `hf_llm` from `database_agent.py`")

st.markdown("### 1) Upload a CSV")
file = st.file_uploader("Drag & drop or browse", type=["csv"])
sep = st.text_input("Separator (optional)", value=",")
encoding = st.text_input("Encoding (optional)", value="utf-8")

df = None
if file:
    with st.spinner("Reading CSV…"):
        df, preview = load_csv(file.getvalue(), sep, encoding)
    st.success(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    with st.expander("Preview (first rows)"):
        st.dataframe(preview, use_container_width=True)

st.markdown("### 2) Ask a question")
default_q = "Filter to July 2020 and report the total hospitalizedIncrease for Texas and also the total across all states."
question = st.text_area("Natural-language question", value=default_q, height=90)

run = st.button("Run query", type="primary", disabled=(df is None or not question))

if run:
    if not allow_dangerous:
        st.error("The Pandas agent requires Python execution. Enable it in the sidebar.")
    elif df is None:
        st.error("Please upload a CSV first.")
    else:
        try:
            llm = get_llm()

            # ---- build the agent on-the-fly with your existing LLM ----
            agent = create_pandas_dataframe_agent(
                llm=llm,
                df=df,
                verbose=False,                    # keep the console clean
                allow_dangerous_code=True,        # required by this agent type
                include_df_in_prompt=False,       # prevents long prompts on big CSVs
                number_of_head_rows=0,
                max_iterations=max_iterations,
                max_execution_time=max_exec_time,
                early_stopping_method="generate",
                agent_executor_kwargs={"handle_parsing_errors": True},
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                return_intermediate_steps=False,
            )

            with st.spinner("Thinking…"):
                t0 = time.time()
                # Keep decoding deterministic by default (relies on your existing hf_llm config)
                result = agent.invoke({"input": question}, config={"callbacks": []})
                elapsed = time.time() - t0

            output = result["output"] if isinstance(result, dict) and "output" in result else result
            st.markdown("### 3) Answer")
            st.write(output)
            st.caption(f"Took {elapsed:.1f}s • Agent iterations ≤ {max_iterations}")

        except Exception as e:
            st.error(f"Run failed:\n\n{e}")
            if import_error:
                with st.expander("Import trace (database_agent.py)"):
                    st.code(import_error, language="text")
