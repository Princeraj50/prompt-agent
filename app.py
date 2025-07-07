import gradio as gr
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os

# 🌱 Load environment
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 🧠 Model
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# 📦 LangGraph State Schema
class PromptState(TypedDict):
    messages: List[BaseMessage]
    latest_prompt: str

# 🔧 Refinement node
def refine_prompt(state: PromptState) -> PromptState:
    prompt = state["messages"][-1].content
    clarity_check = llm.invoke([
        HumanMessage(content=(
            f"Improve the following prompt to make it clearer, more detailed, and copy-paste ready:\n\n'{prompt}'"
        ))
    ])
    improved = clarity_check.content
    return {
        "messages": [AIMessage(content=improved)],
        "latest_prompt": improved
    }

# 🧠 Internal critique node (silent)
def critique_optimizer_output(state: PromptState) -> PromptState:
    _ = llm.invoke([
        HumanMessage(content=(
            f"Critique this instruction:\n\n'{state['latest_prompt']}'\n"
            "Silently fix any ambiguity or formatting issues."
        ))
    ])
    return state

# ✅ Final output node
def final_agent(state: PromptState) -> PromptState:
    final_prompt = state["latest_prompt"]
    suggestion = llm.invoke([
        HumanMessage(content=(
            f"Prompt:\n'{final_prompt}'\n\n"
            "Now give one practical suggestion to improve it further (e.g. add budget, travel method, or dates)."
        ))
    ])
    final = (
        f"{final_prompt}\n\n---\n💡 Suggestion:\n{suggestion.content.strip()}"
    )
    return {
        "messages": [AIMessage(content=final)],
        "latest_prompt": final_prompt
    }

# 🧱 LangGraph setup
graph = StateGraph(PromptState)
for i in range(1, 4):
    graph.add_node(f"refine_{i}", refine_prompt)
    graph.add_node(f"criticize_{i}", critique_optimizer_output)
graph.add_node("final_agent", final_agent)

graph.set_entry_point("refine_1")
graph.add_edge("refine_1", "criticize_1")
graph.add_edge("criticize_1", "refine_2")
graph.add_edge("refine_2", "criticize_2")
graph.add_edge("criticize_2", "refine_3")
graph.add_edge("refine_3", "criticize_3")
graph.add_edge("criticize_3", "final_agent")
graph.add_edge("final_agent", END)

flow = graph.compile()

# 🎨 Gradio interface
def optimize_prompt(user_input: str) -> str:
    result = flow.invoke({
        "messages": [HumanMessage(content=user_input)],
        "latest_prompt": user_input
    })
    return result["messages"][-1].content.strip()

demo = gr.Interface(
    fn=optimize_prompt,
    inputs=gr.Textbox(lines=3, placeholder="Enter a vague or raw prompt..."),
    outputs=gr.Textbox(label="Final Output (Copy-Paste Ready)", lines=8),
    title="🛠️ Prompt Optimizer",
    description="Generate clear, specific, and copyable prompts for general-purpose GenAI tasks."
)

if __name__ == "__main__":
    demo.launch()
