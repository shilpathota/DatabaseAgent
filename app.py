# app.py

# --- top of app.py ---
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"  # avoid inotify limit
os.environ["WATCHDOG_FORCE_POLLING"] = "true"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
import time
import pandas as pd
import streamlit as st
import re

# HF / Torch
import torch
torch.set_num_threads(1)  # or 2 on a 2-core instance

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# LangChain
from langchain_huggingface import HuggingFacePipeline
from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


# ============ UI CONFIG ============
st.set_page_config(page_title="Chat with your CSV", page_icon="📊", layout="wide")
st.title("📊 Chat with your CSV (LangChain + HF on CPU)")

with st.sidebar:
    st.header("Model")
    st.caption("Small CPU model recommended for Streamlit Cloud / basic CPUs.")
    MODEL_ID = st.text_input(
        "HF model id",
        value="Qwen/Qwen2.5-0.5B-Instruct",
        help="Examples: Qwen/Qwen2.5-0.5B-Instruct, TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    st.header("Agent settings")
    allow_dangerous = st.checkbox(
        "Allow Python execution (required by Pandas agent)",
        value=True,
        help="The Pandas DataFrame agent uses a Python REPL tool under the hood."
    )
    # max_new_tokens = st.slider("Max new tokens (answer length)", 16, 512, 64, 16)
    max_iterations = st.slider("Max agent iterations", 1, 8, 3, 1)
    max_exec_time = st.slider("Max execution time (sec)", 5, 90, 30, 5)

    st.header("Decoding")
    deterministic = st.checkbox(
        "Deterministic (recommended)", value=True,
        help="Turns OFF sampling so the agent formats tool calls reliably."
    )
    temperature = st.slider("Temperature (used only if sampling is ON)", 0.0, 1.2, 0.7, 0.1)

    st.info(
        "Tip: If you see PyTorch extension errors on rerun, keep this page's "
        "model cached and avoid changing the model frequently."
    )


# ============ CACHED HELPERS ============
@st.cache_resource(show_spinner=False)
def load_llm(model_id: str, deterministic: bool, temperature: float):
    """
    Build tokenizer/model/pipeline ONCE per process or when any of the
    (model_id, max_new_tokens, deterministic/temperature) params change.
    """
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=False,  # safer; most chat-tuned small models work fine
    )
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=512,
        do_sample=not deterministic,
        **({} if deterministic else {"temperature": float(temperature)})
    )
    return HuggingFacePipeline(pipeline=gen)


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes, sep: str, encoding: str, preview_rows: int = 300):
    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, encoding=encoding)
    # light normalization to make date queries simpler
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df, df.head(min(preview_rows, len(df)))

from typing import Any, Dict, Tuple, List

def extract_final_answer(result: Dict[str, Any]) -> str:
    # 1) Prefer explicit 'FINAL: ...' if your prompt/suffix produces it
    text = result.get("output", "") if isinstance(result, dict) else str(result)
    m = re.search(r"(?im)^\s*FINAL:\s*(.+)$", text)
    if m:
        return m.group(1).strip()

    # 2) Otherwise take the last tool observation (not LLM tokens)
    steps: List[Tuple[Any, Any]] = result.get("intermediate_steps", []) if isinstance(result, dict) else []
    for _action, observation in reversed(steps):
        obs = str(observation).strip()
        if obs:
            # "Trim" = pick the first non-empty line so you get a short, clean answer
            return obs.splitlines()[0].strip()

    # 3) Fallback: last non-empty line of the LLM text
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()

    return text.strip()


# ============ LLM (cached) ============
try:
    hf_llm = load_llm(MODEL_ID, deterministic, temperature)
except Exception as e:
    st.error(f"LLM load failed for `{MODEL_ID}`:\n\n{e}")
    st.stop()


# ============ CSV UPLOAD ============
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


# ============ QUESTION & RUN ============
st.markdown("### 2) Ask a question")
default_q = (
    "Filter to July 2020 and report the total hospitalizedIncrease "
    "for Texas (state == 'TX') and also the total across all states."
)
question = st.text_area("Natural-language question", value=default_q, height=90)

run = st.button("Run query", type="primary", disabled=(df is None or not question))

if run:
    if not allow_dangerous:
        st.error("The Pandas agent requires Python execution. Enable it in the sidebar.")
    elif df is None:
        st.error("Please upload a CSV first.")
    else:
        try:
            agent = create_pandas_dataframe_agent(
                llm=hf_llm,
                df=df,
                verbose=False,
                allow_dangerous_code=True,
                include_df_in_prompt=False,          # allowed because we removed prefix/suffix
                number_of_head_rows=0,
                max_iterations=int(max_iterations),
                max_execution_time=int(max_exec_time),
                early_stopping_method="generate",
                agent_executor_kwargs={"handle_parsing_errors": True},
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                return_intermediate_steps=False,
            )

            with st.spinner("Thinking…"):
                t0 = time.time()
                result = agent.invoke(
                    {"input": question},
                    config={"run_name": "csv_query", "tags": ["ui"], "configurable": {"max_new_tokens": 512}}
                )
                elapsed = time.time() - t0

            result = agent.invoke({"input": question}, config={"callbacks": []})
            text = result["output"] if isinstance(result, dict) else str(result)
            # show just the first non-empty line as the answer
            first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
            st.markdown("### 3) Answer")
            st.write(first)
        except Exception as e:
            st.error(f"Run failed:\n\n{e}")
            # Helpful hint for common causes without breaking flow
            st.info(
                "If this persists, try: "
                "1) keeping Deterministic ON, "
                "2) reducing Max new tokens, "
                "3) confirming the CSV has the expected columns, "
                "4) using a tiny model like TinyLlama for faster CPU responses."
            )
