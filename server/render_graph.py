from langchain_core.runnables.graph import MermaidDrawMethod
from app import graph_instance

graph_instance.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.PYPPETEER, output_file_path="workflow_graph.png"
)

print("✅ Graph rendered and saved as 'workflow_graph.png'")
