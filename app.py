import os
from dotenv import load_dotenv
from scrapegraphai.graphs import SmartScraperGraph
from scrapegraphai.utils import prettify_exec_info
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import gradio as gr
import subprocess
import json
import re

# Ensure Playwright installs required browsers and dependencies
subprocess.run(["playwright", "install"])
subprocess.run(["playwright", "install-deps"])

# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = "##################"

# Initialize the model instances
#repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

llm_model_instance = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN, task="text-generation" 
)

embedder_model_instance = HuggingFaceInferenceAPIEmbeddings(
    api_key=HUGGINGFACEHUB_API_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2"
)

graph_config = {
    "llm": {
        "model_instance": llm_model_instance,
        "model_tokens": 100000, 
    },
    "embeddings": {"model_instance": embedder_model_instance}
}
#######
def clean_json_string(json_str):
    """
    Removes any comments or prefixes before the actual JSON content.
    Returns the cleaned JSON string.
    """
    # Find the first occurrence of '{'
    json_start = json_str.find('{')
    if json_start == -1:
        # If no '{' is found, try with '[' for arrays
        json_start = json_str.find('[')
        if json_start == -1:
            return json_str  # Return original if no JSON markers found
    
    # Extract everything from the first JSON marker
    cleaned_str = json_str[json_start:]
    
    # Verify it's valid JSON
    try:
        json.loads(cleaned_str)
        return cleaned_str
    except json.JSONDecodeError:
        return json_str  # Return original if cleaning results in invalid JSON

def scrape_and_summarize(prompt, source):
    smart_scraper_graph = SmartScraperGraph(
        prompt=prompt,
        source=source,
        config=graph_config
    )
    result = smart_scraper_graph.run()
    
    # Clean the result if it's a string
    if isinstance(result, str):
        result = clean_json_string(result)
    
    exec_info = smart_scraper_graph.get_execution_info()
    return result, prettify_exec_info(exec_info)



#######
def scrape_and_summarize(prompt, source):
     smart_scraper_graph = SmartScraperGraph(
        prompt=prompt,
        source=source,
         config=graph_config
     )
     result = smart_scraper_graph.run()
     exec_info = smart_scraper_graph.get_execution_info()
     return result, prettify_exec_info(exec_info)

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Scrape websites, no-code version")
    gr.Markdown("""
Easily scrape and summarize web content using advanced AI models on the Hugging Face Hub without writing any code. Input your desired prompt and source URL to get started.

This is a no-code version of the excellent library [ScrapeGraphAI](https://github.com/VinciGit00/Scrapegraph-ai).

It's a basic demo and a work in progress. Please contribute to it to make it more useful!

*Note: You might need to add "Output only the results; do not add any comments or include 'JSON OUTPUT' or similar phrases" in your prompt to ensure the LLM only provides the result.*
""")
    with gr.Row():
        with gr.Column():
            
            model_dropdown = gr.Textbox(label="Model", value="Qwen/Qwen2.5-72B-Instruct")
            prompt_input = gr.Textbox(label="Prompt", value="List all the press releases with their headlines and urls. Output only the results; do not add any comments or include 'JSON OUTPUT' or similar phrases.")
            source_input = gr.Textbox(label="Source URL", value="https://www.whitehouse.gov/")
            scrape_button = gr.Button("Scrape and Summarize")
        
        with gr.Column():
            result_output = gr.JSON(label="Result")
            exec_info_output = gr.Textbox(label="Execution Info")

    scrape_button.click(
        scrape_and_summarize,
        inputs=[prompt_input, source_input],
        outputs=[result_output, exec_info_output]
    )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()