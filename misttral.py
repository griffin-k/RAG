import os
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.tools import tool
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup
from mistralai import Mistral

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Configuration
db_path = "web_pages_db"
embeddings_model = "mistral-embed"
llm_model = "mistral-llm"  # Use the appropriate model name for Mistral LLM
mistral_api_key = os.getenv("MISTRAL_API_KEY")
webpage_urls = [
    "https://lgu.edu.pk/cs-faculty/",
]

# Initialize Mistral client
client = Mistral(api_key=mistral_api_key)

# Function to scrape content from a list of web pages
def scrape_web_pages(urls):
    documents = []
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract text from the HTML content
            text = soup.get_text()
            documents.append(text)
        except Exception as e:
            logging.error(f"Error scraping {url}: {e}")
    return documents

# Define the function to process web pages and store in the database
def process_web_pages(urls):
    try:
        # Scrape content from web pages
        documents = scrape_web_pages(urls)

        # Convert raw text into Document objects
        docs = [Document(page_content=text) for text in documents]

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
        split_docs = text_splitter.split_documents(docs)

        # Create embeddings and vector database
        embeddings = client.embeddings.create(model=embeddings_model, inputs=[doc.page_content for doc in split_docs])
        db = FAISS.from_documents(split_docs, embeddings)
        db.save_local(db_path)
        logging.info("Web pages processed and database created.")
    except Exception as e:
        logging.error("Error processing web pages:", exc_info=e)

# Define rephrase query tool
def rephrase_query_tool(query):
    response = client.llm.create(
        model=llm_model,
        inputs=[f"Rephrase the following question: {query}"]
    )
    return response['choices'][0]['text'].strip()

# Define similarity search tool
def similarity_search_tool(query):
    embeddings = client.embeddings.create(model=embeddings_model, inputs=[query])
    db = FAISS.load_local(db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    context = db.similarity_search(query, k=5)
    return context

# Create tools
rephrase_tool = Tool(
    name="rephrase_query",
    description="Rephrase the input question.",
    func=rephrase_query_tool,
)

search_tool = Tool(
    name="similarity_search",
    description="Search the document database for relevant information.",
    func=similarity_search_tool,
)

tools = [rephrase_tool, search_tool]

# Define the prompt template for the agent
template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [ {tool_names} ]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

# Create a PromptTemplate
prompt = PromptTemplate(
    template=template,
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools']
)

# Create an agent
agent = create_react_agent(client.llm, tools, prompt)

# Create an agent executor
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

# Execute a query
def main():
    # Process the web pages and create the database if it doesn't exist
    if not os.path.exists(db_path):
        process_web_pages(webpage_urls)

    query = "who is Dr muhammad asif"
    response = agent_executor.invoke(
        {"input": query}
    )

    print("Response:", response['output'])

if __name__ == "__main__":
    main()
