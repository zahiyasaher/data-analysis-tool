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

# ------------------  QueryUnderstandingTool ---------------------------
def QueryUnderstandingTool(query: str) -> bool:
    messages = [
        {"role": "system", "content": "detailed thinking off. You are an assistant that determines if a query is requesting a data visualization. Respond with only 'true' if the query is asking for a plot, chart, graph, or any visual representation of data. Otherwise, respond with 'false'."},
        {"role": "user", "content": query}
    ]
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=messages,
        temperature=0.1,
        max_tokens=5
    )
    intent_response = response.choices[0].message.content.strip().lower()
    return intent_response == "true"

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
def CodeGenerationAgent(query: str, df: pd.DataFrame):
    should_plot = QueryUnderstandingTool(query)
    tool_used = "PlotCodeGeneratorTool" if should_plot else "CodeWritingTool"
    prompt = PlotCodeGeneratorTool(df.columns.tolist(), query) if should_plot else CodeWritingTool(df.columns.tolist(), query)

    messages = [
        {"role": "system", "content": "detailed thinking off. You are a Python data-analysis expert..."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=messages,
        temperature=0.2,
        max_tokens=1024
    )
    full_response = response.choices[0].message.content
    code = extract_first_code_block(full_response)
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
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=[
            {"role": "system", "content": "detailed thinking on. You are an insightful data analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1024,
        stream=True
    )
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
        response = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
            messages=[
                {"role": "system", "content": "detailed thinking off. You are a data analyst providing brief, focused insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512
        )
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

    left, right = st.columns([3, 7])

    with left:
        st.header("Data Analysis Agent")
        st.markdown("<medium>Powered by NVIDIA Llama-3.1-Nemotron-Ultra-253B-v1</medium>", unsafe_allow_html=True)
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
                    st.markdown(msg["content"], unsafe_allow_html=True)
                    if msg.get("plot_index") is not None:
                        idx = msg["plot_index"]
                        if 0 <= idx < len(st.session_state.plots):
                            st.pyplot(st.session_state.plots[idx], use_container_width=False)

        if file:
            if user_q := st.chat_input("Ask about your data…"):
                st.session_state.messages.append({"role": "user", "content": user_q})
                with st.spinner("Working …"):
                    code, should_plot_flag, code_thinking = CodeGenerationAgent(user_q, st.session_state.df)
                    result_obj = ExecutionAgent(code, st.session_state.df, should_plot_flag)
                    raw_thinking, reasoning_txt = ReasoningAgent(user_q, result_obj)
                    reasoning_txt = reasoning_txt.replace("`", "")
                is_plot = isinstance(result_obj, (plt.Figure, plt.Axes))
                plot_idx = None
                if is_plot:
                    fig = result_obj.figure if isinstance(result_obj, plt.Axes) else result_obj
                    st.session_state.plots.append(fig)
                    plot_idx = len(st.session_state.plots) - 1
                    header = "Here is the visualization you requested:"
                    result_display = ""  # Plot will be rendered separately
                elif isinstance(result_obj, pd.Series):
                    header = "Here is the result:"
                    result_display = f"```\n{result_obj.to_string()}\n```"
                elif isinstance(result_obj, pd.DataFrame):
                    header = f"Result: {len(result_obj)} rows"
                    result_display = f"```\n{result_obj.to_string(index=False)}\n```"
                elif isinstance(result_obj, list):
                    header = "Here is the list you requested:"
                    result_display = f"```\n{', '.join(str(item) for item in result_obj)}\n```"
                else:
                    header = "Here is the result:"
                    result_display = f"```\n{str(result_obj)}\n```"

                # Optional reasoning section
                thinking_html = ""
                if raw_thinking:
                    thinking_html = (
                        '<details class="thinking">'
                        '<summary>🧠 Reasoning</summary>'
                        f'<pre>{raw_thinking}</pre>'
                        '</details>'
                    )

                # Optional explanation/code sections
                explanation_html = reasoning_txt or ""
                code_html = (
                    '<details class="code">'
                    '<summary>View code</summary>'
                    '<pre><code class="language-python">'
                    f'{code}'
                    '</code></pre>'
                    '</details>'
                )

                # Final assistant message
                assistant_msg = f"{thinking_html}<h4>{header}</h4>\n\n{explanation_html}\n\n{code_html}\n\n{result_display}"

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_msg,
                    "plot_index": plot_idx
                })
                st.rerun()

    with st.sidebar:
        st.subheader("🧠 Agent Memory")
        mem = st.session_state.memory.get_summary()
        for item in mem:
            st.markdown(f"- **{item['timestamp']}**: {item['query']} → *{item['tool']}*")

if __name__ == "__main__":
    main()
