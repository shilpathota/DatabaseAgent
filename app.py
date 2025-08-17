# app.py
# Chat with your CSV on CPU: NL -> DuckDB SQL -> Answer
# ----------------------------------------------------
# - Runs on CPU with a tiny HF model (default TinyLlama-1.1B-Chat).
# - Upload a CSV. The file becomes DuckDB table `t`.
# - Ask a natural-language question. The model outputs a single SQL SELECT.
# - We validate & run the SQL in DuckDB and show the result.
#
# How to run:
#   pip install -r requirements.txt
#   streamlit run app.py

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"  # avoid heavy torchvision import
import re
import io
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import duckdb
import streamlit as st

import torch
torch.set_num_threads(1)  # stay light on CPU

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# ------------------------------
# Utilities
# ------------------------------
def infer_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort conversion of columns that look like dates."""
    for c in df.columns:
        lc = c.lower()
        if "date" in lc or "time" in lc or lc.endswith("_dt"):
            try:
                df[c] = pd.to_datetime(df[c], errors="ignore")
            except Exception:
                pass
    return df


def df_schema(df: pd.DataFrame) -> str:
    """Stringify DataFrame schema for prompt."""
    parts = []
    for c, dt in zip(df.columns, df.dtypes):
        parts.append(f"{c} ({str(dt)})")
    return ", ".join(parts)


def extract_sql(text: str) -> str:
    """Extract a SQL statement from model text output."""
    if not text:
        return ""
    # Prefer fenced code blocks ```sql ... ```
    m = re.search(r"```sql\s*(.*?)```", text, flags=re.I | re.S)
    if m:
        return m.group(1).strip().rstrip(";")
    # Any fenced code block
    m = re.search(r"```\s*(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip().rstrip(";")
    # Otherwise, try to capture from SELECT/WITH to end/semicolon
    m = re.search(r"\b(with|select)\b.*", text, flags=re.I | re.S)
    if m:
        candidate = m.group(0).strip()
        # stop at first trailing code fence if any
        candidate = candidate.split("```")[0]
        # cut after last semicolon if present
        if ";" in candidate:
            candidate = candidate.split(";")[0]
        return candidate.strip().rstrip(";")
    return text.strip().rstrip(";")


def is_safe_sql(sql: str) -> tuple[bool, str]:
    """Very simple safety check: allow only SELECT/WITH and disallow DDL/DML keywords."""
    s = sql.strip().lower()
    if not s:
        return False, "Empty SQL."
    if not (s.startswith("select") or s.startswith("with")):
        return False, "Only SELECT/WITH queries are allowed."
    banned = ["insert", "update", "delete", "merge", "drop", "alter", "create", "truncate", "attach", "copy", "pragma"]
    if any(b in s for b in banned):
        return False, "Query contains a disallowed keyword."
    # Must reference our table alias/name `t` (registered from the CSV)
    if " t " not in f" {s} " and " t." not in s and " from t" not in s:
        return False, "Query must reference the table `t`."
    return True, ""


@st.cache_resource(show_spinner=False)
def load_llm(model_id: str, max_new_tokens: int, deterministic: bool, temperature: float):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=int(max_new_tokens),
        do_sample=not deterministic,
        **({} if deterministic else {"temperature": float(temperature)})
    )
    return gen


def build_prompt(df: pd.DataFrame, question: str) -> str:
    schema = df_schema(df)
    examples = f"""
-- Table is named t. Here are a few examples. Output ONLY the SQL.
-- Example 1: Count rows
SELECT COUNT(*) AS row_count FROM t;

-- Example 2: Total of a numeric column (replace colname)
SELECT SUM(colname) AS total FROM t;

-- Example 3: Filter by month (assumes a timestamp/date column named 'date')
SELECT SUM(hospitalizedIncrease) AS total_tx_july_2020
FROM t
WHERE state = 'TX' AND strftime(date, '%Y-%m') = '2020-07';
"""
    sys = f"""You are a senior data analyst. Write one DuckDB SQL query to answer the question.
Rules:
- Use ONLY the table `t`.
- Use ONLY existing columns.
- Prefer ISO date functions like strftime(date, '%Y-%m') for month filters.
- Return aggregated answers with SUM/COUNT/AVG etc. If the question asks for a single number, produce a single-row, single-column result.
- Do not add explanations. Output ONLY the SQL.
Schema: {schema}
{examples}
Question: {question}
SQL:
"""
    return sys


def run_query(df: pd.DataFrame, sql: str, limit_rows: int = 200) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("t", df)
    # Enforce a hard cap by wrapping if no LIMIT present
    s = sql.strip()
    if " limit " not in s.lower():
        s = f"SELECT * FROM ({s}) AS sub LIMIT {int(limit_rows)}"
    out = con.sql(s).df()
    con.close()
    return out


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="CSV SQL Chat (CPU)", page_icon="📊", layout="wide")
st.title("📊 CSV SQL Chat (CPU)")

with st.sidebar:
    st.subheader("Model")
    model_id = st.text_input(
        "HF model id",
        value="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Small CPU-friendly chat model. Other options: Qwen/Qwen2.5-0.5B-Instruct",
    )
    max_new_tokens = st.slider("Max new tokens (answer length)", 32, 256, 128, 16)
    deterministic = st.checkbox("Deterministic (recommended)", value=True)
    temperature = st.slider("Temperature (used only if sampling ON)", 0.0, 1.2, 0.7, 0.1)
    limit_rows = st.slider("Display up to N result rows", 20, 1000, 200, 20)
    st.caption("Runs fully on CPU; keep settings modest for faster responses.")

# lazy-load the LLM so app starts quickly
llm = load_llm(model_id, max_new_tokens, deterministic, temperature)

st.markdown("### 1) Upload CSV")
file = st.file_uploader("Drop a CSV file here", type=["csv"])
sep = st.text_input("Separator (optional)", value=",")
encoding = st.text_input("Encoding (optional)", value="utf-8")

if file:
    df = pd.read_csv(io.BytesIO(file.getvalue()), sep=sep, encoding=encoding)
    df = infer_datetime_columns(df)
    st.success(f"Loaded **{len(df):,}** rows × **{df.shape[1]}** columns")
    with st.expander("Preview (first rows)"):
        st.dataframe(df.head(20), use_container_width=True)
    with st.expander("Columns"):
        st.write(list(df.columns))

    st.markdown("### 2) Ask a question")
    default_q = "Filter to July 2020 and report the total hospitalizedIncrease for Texas and also the total across all states."
    question = st.text_area("Natural-language question", value=default_q, height=90)
    run = st.button("Run", type="primary")

    if run and question.strip():
        with st.spinner("Thinking…"):
            prompt = build_prompt(df, question.strip())
            try:
                out = llm(prompt, return_full_text=False)[0]["generated_text"]
            except TypeError:
                # some pipelines ignore return_full_text
                out = llm(prompt)[0]["generated_text"]
            sql = extract_sql(out)

        ok, why = is_safe_sql(sql)
        with st.expander("Show SQL", expanded=True):
            st.code(sql or "(none)", language="sql")
            if not ok:
                st.error(f"Query rejected: {why}")

        if ok:
            try:
                result = run_query(df, sql, limit_rows=limit_rows)
                st.markdown("### 3) Answer")
                if result.shape == (1, 1) and pd.api.types.is_numeric_dtype(result.dtypes[0]):
                    label = str(result.columns[0])
                    value = float(result.iloc[0, 0])
                    st.metric(label=label, value=f"{value:,.2f}")
                else:
                    st.dataframe(result, use_container_width=True, height=420)
                csv_bytes = result.to_csv(index=False).encode("utf-8")
                st.download_button("Download results (CSV)", data=csv_bytes, file_name="answer.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Execution failed: {e}")
else:
    st.info("Upload a CSV to begin.")
