# PART-1
# #The Brain - selecting a chat model
# LOAD -> SPLIT -> EMBED -> RETRIEVE. AI is asleep in these phases. These phases are handled by my script.
# The LLM does not knows or runs my python code.


import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")



# Injection Engine data->document with same format
# Loading Documents. 
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain.tools import tool
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model

def load_web_documents(urls: list):
    """
    Takes a list of URLs and returns a flat list of LangChain documents.
    """
    print(f"Fetching data from {len(urls)} URLs...")

    nested_docs = [WebBaseLoader(url).load() for url in urls] # becomes a list of lists -- [[Doc1], [Doc2], [Doc3]]

    flat_docs = [doc for sublist in nested_docs for doc in sublist] # flatten the list of lists -- ([Doc1, Doc2, Doc3])

    return flat_docs


# Tool for the langgraph
@tool  # The interface between the custom logic and AI's thinking.
def retrieve_blog_posts(query: str) -> str:  # query is an args as a JSON object.
    """
    Search and return information about Lilian Weng blog posts.
    """
    docs = retriever.invoke(query) # Script sends the JSON schema to OpenAI
    return "\n\n".join([doc.page_content for doc in docs])
    
retriever_tool = retrieve_blog_posts


# THINKING OF AI MODEL HAPPENS.
# Generating Query. Graph node. - # PART-2 - Autonomous Agent - that thinks before acting. The LANGGRAPH LOGIC
# Includes MessagesState makes it easy to use messages

response_model = init_chat_model("gpt-5-nano", temperature=0) # Initialize the model

def generate_query_or_respond(state: MessagesState):
    """
    Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
     
     # We bind the tool with the response_model so that it can decide whether to proceed or use the tool.

    response = (
        response_model.bind_tools([retriever_tool]).invoke(state["messages"]) # .invoke the messages from the langgraph state, we sent a hidden instruction to the OpenAI by using the .bind_tools([retriever_tool]). Returns an AIMessage Object.
    )
    return {"messages": [response]} # Return the new message to be added to the state


# GRADING DOCUMENTS

from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import convert_to_messages

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n"
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

# Pytdantic is like zod for data validation, BaseModel is a class that serves as a foundation for defining data models using python type annotations. Data must be in the specified structure in class GradeDocuments that inherits BaseModel class so that all inputs match this data specified.
class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

grader_model = init_chat_model("gpt-5-nano", temperature=0)

# The router of data that decides where should the data go next
def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content # 1st message is the question
    context = state["messages"][1].content  # 2nd message is the retrieved Text
    
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    
    score = response.binary_score
    # The Edge of the graph
    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question" # Fix the question and search again
    
    


# The execution

if __name__ == "__main__": # python standard only run the code inside here if i run python main.py
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    docs = load_web_documents(urls)
    print(f"Successfully loaded {len(docs)} documents.")
    # if docs:
    #    print(f"Sample Metadata: {docs[0].metadata}") # metadata = source url, title, language.

    # print("RAW DATA CHECK:")
    # print(docs[0].page_content.strip()[:1000])


#Splitting docs

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder( # counts tokens like cat, ing, tion
            chunk_size=100,
            chunk_overlap=50,
        )
    doc_splits = text_splitter.split_documents(docs) # Is Storing chunks of the docs
    print(f"Split into {len(doc_splits)} chunks.")
    # print(doc_splits[0].page_content.strip()) # strips the whole page content we might set a limit like 1000 or even leave like this.
    # print(f"Total chunks created: {len(doc_splits)}")

# Select an embeddigns model, Select a vector store the memory to turn the text into vectors numbers.:

    URI = "milvus_example.db"

    vector_store = Milvus.from_documents(
        documents=doc_splits,
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
        connection_args={"uri": URI},
        drop_old=True,
    )
    # Convert it into the Retriever tool for LangGraph
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # print("Testing Tool...")
    # result = retriever_tool.invoke({"query": "types of reward hacking"})
    # print(f"Reward Hacking Result: {result[:1000]}... {len(result)}\n\n")

    # For testing the Node - 1 - generate_query_or_respond 
    # Can write the content as : "Hello!" or a question about blog posts.
    test_input = {"messages": [{"role": "user", "content": "What does Lilian Weng say about types of reward hacking?",}]} # Faking the state to see if the function works. Must match with MessagesState. This will generate a tool call Object not a text sentence or an answer.
    output = generate_query_or_respond(test_input)

    print(" AI Response Test --- ")
    output["messages"][-1].pretty_print() # We need AI response so [-1] always have AI response at the very end of messages list since langgraph appends messages.
    # .pretty_print() organizes the messy object to a clean AI response.
    
    input = {
        "messages": convert_to_messages(
            [
                {
                    "role": "user",
                    "content": "What does Lilian Wang say about types of reward hacking?",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "retrieve_blog_posts",
                            "args": {"query": "types of reward hacking"},
                        }
                    ],
                },
                {"role": "tool", "content": "meow", "tool_call_id": "1"},
            ]
        )
    }
    
    output = grade_documents(input)
    print(f"Check whether the retrieved docs is relevant to the user question ---\n  {output}")
    
    
    
    

