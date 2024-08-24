import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.tools import tool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Configuration
pdf_path = "test.pdf"
db_path = "al-alim_db"
embeddings_model = "models/embedding-001"
llm_model = "gemini-pro"
google_api_key = "AIzaSyCvxLfNfHTQ7DuniFo4_LG1CoZ07nAWndo"

# Define the function to rephrase the query
def rephrase_query_tool(query):
    rephrasing_prompt_template = PromptTemplate(
        template="I need you to rephrase the following question: {question}",
        input_variables=["question"]
    )
    prompt = rephrasing_prompt_template.format(question=query)
    llm = ChatGoogleGenerativeAI(model=llm_model, google_api_key=google_api_key)
    response = llm.invoke(prompt)
    if hasattr(response, "content"):
        return response.content
    else:
        logging.warning("Response object does not have 'content' attribute.")
        return None

# Define the function to perform similarity search
def similarity_search_tool(query):
    # Initialize the LLM and the database
    embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model)
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

# Initialize the language model
llm = ChatGoogleGenerativeAI(model=llm_model, google_api_key=google_api_key)

# Create an agent
agent = create_react_agent(llm, tools, prompt)

# Create an agent executor
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

def main():
    # Check if the database already exists
    if not os.path.exists(db_path):
        # Process the PDF and create the database
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
            docs = text_splitter.split_documents(pages)
            embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model)
            db = FAISS.from_documents(docs, embeddings)
            db.save_local(db_path)
            logging.info("PDF processed and database created.")
        except Exception as e:
            logging.error("Error processing PDF:", exc_info=e)
    else:
        logging.info("Database already exists. Skipping PDF processing.")

    # Execute a query
    query = "who select the Deputy Secretary"
    response = agent_executor.invoke(
        {"input": query}
    )

    print("Response:", response['output'])

if __name__ == "__main__":
    main()
