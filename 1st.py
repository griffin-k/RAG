import os
import json
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEndpoint

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Define paths and model names
db_path = "G0rbit-db"
embeddings_model = "models/embedding-001"
llm_model = "gemini-pro"
google_api_key = "AIzaSyDMYClmwboCjTOxIw-vED1ZLwnz-5MugOs"
json_file_paths = ["a.json","b.json","c.json","d.json","e.json","f.json","g.json","h.json","j.json","k.json"]

def read_json_files(file_paths):
    documents = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                if isinstance(data, list):
                    documents.extend(data)
                else:
                    logging.warning(f"JSON file {file_path} does not contain a list of documents.")
        except Exception as e:
            logging.error(f"Error reading JSON file {file_path}: {e}")
    return documents

def process_json_files(file_paths):
    try:
        documents = read_json_files(file_paths)
        if not documents:
            logging.error("No documents found to process.")
            return

        docs = [Document(page_content=str(text)) for text in documents]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
        split_docs = text_splitter.split_documents(docs)

        if not split_docs:
            logging.error("No documents were split into chunks. Check your input data.")
            return

        embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model)
        logging.info("Generating embeddings...")
        try:
            test_doc = split_docs[0].page_content
            test_embedding = embeddings.embed_documents([test_doc])
            if not test_embedding or not isinstance(test_embedding[0], list):
                raise ValueError("Embeddings generation failed or returned invalid format.")
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            return

        db = FAISS.from_documents(split_docs, embeddings)
        db.save_local(db_path)
        logging.info("JSON files processed and database created.")
    except Exception as e:
        logging.error("Error processing JSON files:", exc_info=e)

def similarity_search_tool(query):
    embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model)
    if not os.path.exists(db_path):
        logging.error(f"Database path {db_path} does not exist.")
        return None

    try:
        index_path = os.path.join(db_path, "index.faiss")
        if not os.path.isfile(index_path):
            logging.error(f"FAISS index file {index_path} does not exist.")
            return None
        
        db = FAISS.load_local(db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        context = db.similarity_search(query, k=5)
        return context
    except Exception as e:
        logging.error(f"Error loading FAISS index: {e}")
        return None

def rephrase_query_tool(query):
    rephrasing_prompt_template = PromptTemplate(
        template="I need you to rephrase the following question: {question}",
        input_variables=["question"]
    )
    prompt = rephrasing_prompt_template.format(question=query)
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        token=os.getenv('HUGGINGFACE_API_TOKEN')
    )
    try:
        response = llm.invoke(prompt)
        
        # Log the response for debugging
        logging.info(f"Rephrase query response: {response}")

        # Handle response as a string directly
        if isinstance(response, str):
            return response
        else:
            logging.warning("Response is not a string. Check the response format.")
            return None
    except Exception as e:
        logging.error(f"Error rephrasing query: {e}")
        return None



# Define tools
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

# Define prompt template for the agent
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

# Create prompt template
prompt = PromptTemplate(
    template=template,
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools']
)

# Create agent and executor
llm = ChatGoogleGenerativeAI(model=llm_model, google_api_key=google_api_key)
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=False, handle_parsing_errors=True
)

def main():
    if not os.path.exists(db_path):
        process_json_files(json_file_paths)

    query = "how are you"
    try:
        response = agent_executor.invoke({"input": query})
        print("Response:", response['output'])
    except Exception as e:
        logging.error(f"Error during agent execution: {e}")

if __name__ == "__main__":
    main()
