# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, io, re, logging
import pandas as pd
import streamlit as st
from openai import OpenAI
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# === Logging Setup ===
logging.basicConfig(
    filename="agent_logs.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# === Memory Class ===
class AgentMemory:
    def __init__(self):
        self.history = []

    def add(self, query: str, tool: str, code: str, result: Any):
        entry = {
            "timestamp": str(datetime.now()),
            "query": query,
            "tool": tool,
            "code": code,
            "result_preview": str(result)[:200]
        }
        self.history.append(entry)
        logging.info(f"Logged Query → Tool: {tool}\nQuery: {query}\nCode: {code}\n")

    def get_summary(self):
        return self.history[-5:]

# === Configuration ===
api_key = os.environ.get("NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Model configuration with fallback
DEFAULT_MODEL = os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
FALLBACK_MODEL = os.environ.get("NVIDIA_FALLBACK_MODEL", "mistralai/mixtral-8x7b-instruct")


def _should_try_fallback(exc: Exception) -> bool:
    text = str(exc)
    return ("404" in text) or ("Not Found" in text) or ("Function id" in text)


def create_chat_completion(messages: List[Dict[str, Any]], temperature: float = 0.2, max_tokens: int = 1024, stream: bool = False):
    """Try the configured model; on 404/Not Found, try the fallback model.
    Returns (response, used_model). Raises the last exception if all fail.
    """
    models_to_try: List[str] = []
    if DEFAULT_MODEL:
        models_to_try.append(DEFAULT_MODEL)
    if FALLBACK_MODEL and FALLBACK_MODEL not in models_to_try:
        models_to_try.append(FALLBACK_MODEL)

    last_exc: Exception | None = None
    for model_name in models_to_try:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            return resp, model_name
        except Exception as exc:  # broad: SDK raises various subclasses across versions
            logging.error(f"Chat completion failed for model '{model_name}': {exc}")
            last_exc = exc
            # Only continue to fallback for specific not-found style errors
            if _should_try_fallback(exc):
                continue
            else:
                break
    # If we reach here, all attempts failed
    if last_exc:
        raise last_exc
    raise RuntimeError("No model configured for chat completion")

# ------------------  QueryUnderstandingTool ---------------------------
def QueryUnderstandingTool(query: str) -> bool:
    messages = [
        {"role": "system", "content": "detailed thinking off. You are an assistant that determines if a query is requesting a data visualization. Respond with only 'true' if the query is asking for a plot, chart, graph, or any visual representation of data. Otherwise, respond with 'false'."},
        {"role": "user", "content": query}
    ]
    try:
        response, used_model = create_chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=5,
            stream=False,
        )
        # track model in session, if available
        try:
            st.session_state["model_name"] = used_model
        except Exception:
            pass
        intent_response = response.choices[0].message.content.strip().lower()
        return intent_response == "true"
    except Exception as exc:
        logging.warning(f"QueryUnderstandingTool failed; using heuristic. Error: {exc}")
        # Heuristic fallback if model is unavailable
        return bool(re.search(r"\b(plot|chart|graph|visual|scatter|bar|line|histogram|pie)\b", query, re.IGNORECASE))

# ------------------  PlotCodeGeneratorTool ---------------------------
def PlotCodeGeneratorTool(cols: List[str], query: str) -> str:
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code using pandas **and matplotlib** (as plt) to answer:
    "{query}"

    Rules
    -----
    1. Use pandas for data manipulation and matplotlib.pyplot (as plt) for plotting.
    2. Assign the final result (DataFrame, Series, scalar *or* matplotlib Figure) to a variable named `result`.
    3. Create only ONE relevant plot. Set `figsize=(6,4)`, add title/labels.
    4. Return your answer inside a single markdown fence that starts with ```python and ends with ```.
    """

# ------------------  CodeWritingTool ---------------------------------
def CodeWritingTool(cols: List[str], query: str) -> str:
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code (pandas **only**, no plotting) to answer:
    "{query}"

    Rules
    -----
    1. Use pandas operations on `df` only.
    2. Assign the final result to `result`.
    3. Wrap the snippet in a single ```python code fence (no extra prose).
    """

# === CodeGenerationAgent ==============================================
def CodeGenerationAgent(query: str, df: pd.DataFrame, chat_context: List[Dict[str, str]] | None = None):
    should_plot = QueryUnderstandingTool(query)
    tool_used = "PlotCodeGeneratorTool" if should_plot else "CodeWritingTool"
    base_prompt = PlotCodeGeneratorTool(df.columns.tolist(), query) if should_plot else CodeWritingTool(df.columns.tolist(), query)

    context_text = ""
    if chat_context:
        recent_ctx = chat_context[-6:]
        ctx_lines: List[str] = []
        for entry in recent_ctx:
            etype = entry.get("type")
            content = entry.get("content", "")
            if etype == "user":
                ctx_lines.append(f"User: {content}")
            elif etype == "code":
                ctx_lines.append("Previous code:\n" + content[:2000])
            elif etype == "explanation":
                ctx_lines.append("Previous explanation:\n" + content[:800])
        if ctx_lines:
            context_text = "Context for refinement (use to improve the next answer):\n" + "\n\n".join(ctx_lines)

    user_prompt = (context_text + "\n\n" + base_prompt) if context_text else base_prompt

    messages = [
        {"role": "system", "content": "detailed thinking off. You are a Python data-analysis expert..."},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response, used_model = create_chat_completion(
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
            stream=False,
        )
        try:
            st.session_state["model_name"] = used_model
        except Exception:
            pass
        full_response = response.choices[0].message.content
        code = extract_first_code_block(full_response)
    except Exception as exc:
        logging.error(f"CodeGenerationAgent failed; falling back to df.describe(). Error: {exc}")
        code = "result = df.describe()"
    st.session_state.memory.add(query, tool_used, code, "(Pending execution)")
    return code, should_plot, ""

# === ExecutionAgent ====================================================
def ExecutionAgent(code: str, df: pd.DataFrame, should_plot: bool):
    env = {"pd": pd, "df": df}
    if should_plot:
        plt.rcParams["figure.dpi"] = 100
        env["plt"] = plt
        env["io"] = io
    try:
        exec(code, {}, env)
        result = env.get("result", None)
    except Exception as exc:
        logging.error(f"Execution failed: {exc}\nCode:\n{code}")
        fallback_code = "result = df.describe()"
        logging.info("Fallback to: df.describe()")
        try:
            exec(fallback_code, {}, env)
            result = env.get("result", None)
        except Exception as exc2:
            result = f"Final error executing fallback code: {exc2}"
            logging.critical(f"Fallback failed: {exc2}")
    return result

# === ReasoningCurator TOOL =========================================
def ReasoningCurator(query: str, result: Any) -> str:
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))
    if is_error:
        desc = result
    elif is_plot:
        title = ""
        if isinstance(result, plt.Figure):
            title = result._suptitle.get_text() if result._suptitle else ""
        elif isinstance(result, plt.Axes):
            title = result.get_title()
        desc = f"[Plot Object: {title or 'Chart'}]"
    else:
        desc = str(result)[:300]

    if is_plot:
        prompt = f'The user asked: "{query}". Below is a description of the plot result: {desc} Explain in 2–3 concise sentences what the chart shows (no code talk).'
    else:
        prompt = f'The user asked: "{query}". The result value is: {desc} Explain in 2–3 concise sentences what this tells about the data (no mention of charts).'
    return prompt

# === ReasoningAgent (streaming) =========================================
def ReasoningAgent(query: str, result: Any):
    prompt = ReasoningCurator(query, result)
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))
    try:
        response, used_model = create_chat_completion(
            messages=[
                {"role": "system", "content": "detailed thinking on. You are an insightful data analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024,
            stream=True,
        )
        try:
            st.session_state["model_name"] = used_model
        except Exception:
            pass
    except Exception as exc:
        logging.error(f"ReasoningAgent failed: {exc}")
        return "", "I generated the result but could not produce an explanation due to a model error."

    thinking_placeholder = st.empty()
    full_response = ""
    thinking_content = ""
    in_think = False
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            full_response += token
            if "<think>" in token:
                in_think = True
                token = token.split("<think>", 1)[1]
            if "</think>" in token:
                token = token.split("</think>", 1)[0]
                in_think = False
            if in_think or ("<think>" in full_response and not "</think>" in full_response):
                thinking_content += token
                thinking_placeholder.markdown(
                    f'<details class="thinking" open><summary>🤔 Model Thinking</summary><pre>{thinking_content}</pre></details>',
                    unsafe_allow_html=True
                )
    cleaned = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
    return thinking_content, cleaned

# === DataFrameSummary TOOL =========================================
def DataFrameSummaryTool(df: pd.DataFrame) -> str:
    prompt = f"""
        Given a dataset with {len(df)} rows and {len(df.columns)} columns:
        Columns: {', '.join(df.columns)}
        Data types: {df.dtypes.to_dict()}
        Missing values: {df.isnull().sum().to_dict()}
        Provide:
        1. A brief description of what this dataset contains
        2. 3-4 possible data analysis questions that could be explored
        Keep it concise and focused."""
    return prompt

# === DataInsightAgent ===============================================
def DataInsightAgent(df: pd.DataFrame) -> str:
    prompt = DataFrameSummaryTool(df)
    try:
        response, used_model = create_chat_completion(
            messages=[
                {"role": "system", "content": "detailed thinking off. You are a data analyst providing brief, focused insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512,
            stream=False,
        )
        try:
            st.session_state["model_name"] = used_model
        except Exception:
            pass
        return response.choices[0].message.content
    except Exception as exc:
        return f"Error generating dataset insights: {exc}"

# === Helpers ===========================================================
def extract_first_code_block(text: str) -> str:
    start = text.find("```python")
    if start == -1:
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        return ""
    return text[start:end].strip()

# === Main App =========================================================
def main():
    st.set_page_config(layout="wide")
    if "plots" not in st.session_state:
        st.session_state.plots = []
    if "memory" not in st.session_state:
        st.session_state.memory = AgentMemory()
    if "model_name" not in st.session_state:
        st.session_state.model_name = DEFAULT_MODEL
    if "chat_context" not in st.session_state:
        st.session_state.chat_context: List[Dict[str, str]] = []
    if "pending" not in st.session_state:
        st.session_state.pending = None

    left, right = st.columns([3, 7])

    with left:
        st.header("Data Analysis Agent")
        st.markdown(f"<medium>Powered by {st.session_state.model_name}</medium>", unsafe_allow_html=True)
        file = st.file_uploader("Choose CSV", type=["csv"])
        if file:
            if ("df" not in st.session_state) or (st.session_state.get("current_file") != file.name):
                st.session_state.df = pd.read_csv(file, encoding='latin1')
                st.session_state.current_file = file.name
                st.session_state.messages = []
                with st.spinner("Generating dataset insights …"):
                    st.session_state.insights = DataInsightAgent(st.session_state.df)
            st.dataframe(st.session_state.df.head())
            st.markdown("### Dataset Insights")
            st.markdown(st.session_state.insights)
        else:
            st.info("Upload a CSV to begin chatting with your data.")

    with right:
        st.header("Chat with your data")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    # Render side-by-side columns for assistant results
                    if msg["role"] == "assistant" and (msg.get("code") or msg.get("explanation") or msg.get("result_display")):
                        col_code, col_mid, col_expl = st.columns([4, 3, 5])
                        with col_code:
                            if msg.get("code"):
                                st.subheader("Code")
                                st.code(msg["code"], language="python")
                        with col_mid:
                            st.subheader("Result")
                            idx = msg.get("plot_index")
                            if idx is not None and 0 <= idx < len(st.session_state.plots):
                                st.pyplot(st.session_state.plots[idx], use_container_width=False)
                            elif msg.get("result_display"):
                                st.markdown(msg["result_display"], unsafe_allow_html=True)
                        with col_expl:
                            if msg.get("explanation"):
                                st.subheader("Explanation")
                                st.markdown(msg["explanation"], unsafe_allow_html=True)
                    else:
                        st.markdown(msg.get("content", ""), unsafe_allow_html=True)

        if file:
            # Chat input: generate code, but do not execute until user edits/confirms
            if user_q := st.chat_input("Ask about your data…"):
                st.session_state.messages.append({"role": "user", "content": user_q})
                st.session_state.chat_context.append({"type": "user", "content": user_q})
                with st.spinner("Generating code …"):
                    code, should_plot_flag, _ = CodeGenerationAgent(user_q, st.session_state.df, chat_context=st.session_state.chat_context)
                st.session_state.pending = {
                    "query": user_q,
                    "code": code or "result = df.describe()",
                    "should_plot": should_plot_flag,
                }
                st.rerun()

            # If there is pending code, allow editing and running
            if st.session_state.pending is not None:
                st.markdown("---")
                st.subheader("Review and edit the code, then run")
                st.session_state.pending["code"] = st.text_area(
                    "Generated code (editable)",
                    value=st.session_state.pending.get("code", ""),
                    key="code_editor",
                    height=300,
                )
                run_col, rerun_col, _ = st.columns([2, 3, 5])
                run_clicked = run_col.button("Run code")
                rerun_clicked = rerun_col.button("Rerun with changes")
                if run_clicked or rerun_clicked:
                    code_to_run = st.session_state.pending.get("code", "result = df.describe()")
                    with st.spinner("Executing …"):
                        result_obj = ExecutionAgent(code_to_run, st.session_state.df, st.session_state.pending.get("should_plot", False))
                        raw_thinking, reasoning_txt = ReasoningAgent(st.session_state.pending.get("query", ""), result_obj)
                        reasoning_txt = (reasoning_txt or "").replace("`", "")
                    is_plot = isinstance(result_obj, (plt.Figure, plt.Axes))
                    plot_idx = None
                    result_display = ""
                    if is_plot:
                        fig = result_obj.figure if isinstance(result_obj, plt.Axes) else result_obj
                        st.session_state.plots.append(fig)
                        plot_idx = len(st.session_state.plots) - 1
                    elif isinstance(result_obj, pd.Series):
                        result_display = f"```
{result_obj.to_string()}
```"
                    elif isinstance(result_obj, pd.DataFrame):
                        result_display = f"```
{result_obj.to_string(index=False)}
```"
                    elif isinstance(result_obj, list):
                        result_display = f"```
{', '.join(str(item) for item in result_obj)}
```"
                    else:
                        result_display = f"```
{str(result_obj)}
```"

                    # Append assistant message with side-by-side data
                    st.session_state.messages.append({
                        "role": "assistant",
                        "code": code_to_run,
                        "explanation": reasoning_txt,
                        "plot_index": plot_idx,
                        "result_display": result_display,
                    })

                    # Update conversation context for refinement
                    st.session_state.chat_context.append({"type": "code", "content": code_to_run})
                    if reasoning_txt:
                        st.session_state.chat_context.append({"type": "explanation", "content": reasoning_txt})

                    st.rerun()

    with st.sidebar:
        st.subheader("🧠 Agent Memory")
        mem = st.session_state.memory.get_summary()
        for item in mem:
            st.markdown(f"- **{item['timestamp']}**: {item['query']} → *{item['tool']}*")

if __name__ == "__main__":
    main()
