from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
import warnings
from dotenv import load_dotenv
import os
import logging
from termcolor import colored
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph, END
from agents import SpecializedAgent
from prompts import PROMPT_ADMINISTRATIVO, PROMPT_ESTUDIANTE, PROMPT_PROFESOR, PROMPT_EXTERNO

COLLECTION_NAMES = ["Administrativo", "Estudiante", "Profesor", "Externo"]
list_of_prompts = [PROMPT_ADMINISTRATIVO, PROMPT_ESTUDIANTE, PROMPT_PROFESOR, PROMPT_EXTERNO]
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPEN_ROUTER'),
    model="gpt-3.5-turbo"
)

def specialized_node(state):
    user = state["user"]
    query = state["query"]

    logging.info(colored(f"Agente {user} procesando pregunta...", "magenta"))
    
    for collection_name, prompt in zip(COLLECTION_NAMES, list_of_prompts):    
        if user == collection_name:
            specialized_agent = SpecializedAgent(llm, collection_name = collection_name, prompt_template = prompt)
    
    return specialized_agent.run(query, log_context = True)

graph = Graph()
graph.add_node("specialized_node", specialized_node)
graph.add_edge("specialized_node", END)
graph.set_entry_point("specialized_node")
compiled_graph = graph.compile()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for the Flask app

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query = data.get('query')
    user = data.get('user')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    logging.info("Invoking graph...")
    response = compiled_graph.invoke({"query": query, "user": user})
    logging.info(f"Response: {response}")
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)