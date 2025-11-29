#   Written by:  Mark W Kiehl
#   http://mechatronicsolutionsllc.com/
#   http://www.savvysolutions.info/savvycodesolutions/

# MIT License
# Copyright (c) 2025 Mechatronic Solutions LLC

"""
This Python script implements a DSPy-based Multi-Agent Pipeline designed to tackle complex user queries 
by strategically combining information from a local document database (RAG) and the external web. 
The primary goal is to find a complete and accurate answer in the most efficient way possible.

The script leverages DSPy Signatures (which define the input/output schema for LLM tasks) and 
DSPy Modules/Agents (which execute those tasks) to break down the user query into manageable steps.


MasterPlanAgent	        Takes the user's initial, complex query and breaks it down into a list of smaller, targeted master_plan_query steps.

SearchDocsAgent	        Executes a search against the local documents (using a hybrid BM25 and Chroma RAG system) and summarizes the findings for a specific step query.

SearchWebAgent	        Executes a search against the web (using the Tavily tool) and summarizes the findings for a specific step query.

RawContentCheckAgent    Crucial for Early Termination: Immediately analyzes the raw search results against the user's original query to see if the full answer is already present.

ReflectAgent	        Determines if a specific master_plan_query step has been sufficiently answered based on its search history.

FinalizeStepAgent	    Reviews the complete search history for one step and synthesizes a final, factual summary for that step.

FinalizeMasterPlanAgent	Takes all the final step summaries and synthesizes them into a single, cohesive answer to the user's original query.


Below are the PIP INSTALL requirements.  
If using Windows OS, you can use run the file "make_venv.bat" from a Windows Command Prompt and it will create the Python virtual environment and install the required files.

PIP INSTALL:

dspy
openai
python-dotenv
langchain-core
langchain-tavily
langchain-community
langchain-chroma
langchain-openai
langchain-classic
rank_bm25
rich

"""

# Define the script version in terms of Semantic Versioning (SemVer)
# when Git or other versioning systems are not employed.
__version__ = "1.0.2"
from pathlib import Path
print("'" + Path(__file__).stem + ".py'  v" + __version__)
# 1.0.0     New pipeline architecture. 
# 1.0.1     Works.  But it can miss results from a master plan step iteration that answers the user's query. 
# 1.0.2     Add Top-Level Reflector Agent to see if results from any master plan step iteration happens to answer the user's query. 
#           Introduce a dedicated Raw Content Check Agent that runs immediately after any search (local or web) is executed. 
#           This agent directly checks the raw, unsummarized data against the user's overall query to bypass the step-level 
#           agent's decision to ensure the earliest possible termination.
#           Published on medium.com  https://medium.com/@markwkiehl/building-a-multi-agent-rag-system-with-dspy-d4b497475b83

""" 
API Requirements:

Edit the included ".env" file and update the following with your API keys:

OPENAI_API_KEY
TAVILY_API_KEY

"""

if not Path(Path.cwd()).joinpath(".env").is_file(): raise Exception(f"Download the .env template file from GitHub or create one yourself with the required API keys")
from dotenv import load_dotenv
load_dotenv()


import os
import shutil                               
import uuid
import re
import json

import dspy

from langchain_core.documents import Document
from rich.console import Console
from langchain_tavily import TavilySearch
from typing import List, Dict
import pickle
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


config = {
    "llm_provider": "openai",                                           # The LLM provider we are using
    "embedding_model": "text-embedding-3-small",                        # The model for creating document embeddings
    "max_step_iterations": 3,                                           # The maximum number iterations per master plan step. 
    "path_chroma_db": Path(Path.cwd()).joinpath("chroma_langchain_db"), # Folder for the Chroma db vector store
    "chroma_collection_name": "Biography_of_Christopher_Diaz",          # Chroma db collection name
}



# --- Search Local Documents & Utility Functions ---

# --- Search Local Documents & Utility Functions ---


# documents for (fake) biography of Christopher Diaz
def get_docs_christopher_diaz(rebuild_source_files:bool=False, doc_format:str="LangChain", verbose:bool=True) -> List[str]:
    """
    Generates or reads the documents for the Christopher Diaz fake facts.

    If 'rebuild_source_files' is True, rebuilds documents.

    documents = get_docs_christopher_diaz(rebuild_source_files=False)

    USAGE:
    
    documents = get_docs_christopher_diaz()
    print(documents)
    # ['Christopher Diaz was a white man born in Ryandbury, Rhode Island', 'Christopher Diaz worked for Willis Group LLC as a clinical scientist in Lewischester, MI', 'Christopher Diaz married Cindy Lopez in 1965', 'Cindy Lopez gave Christopher Diaz two sons and one daughter', "Christopher Diaz enjoyed being a sports coach for the public school Griffin, where Christopher Diaz's kids attended", 'Christopher Diaz died at the age of 82 in 2014, having been born in either 1931 or 1932.']

    """

    def docs_langchain_fake_christopher_diaz_enriched() -> List[Document]:
        """
        Returns a list of LangChain documents consisting of fake facts about Christopher Diaz.

        These six fake facts about Christopher Diaz have already been optimized for clarity 
        of the relationships, and to state all temporal relationships using savvy_optimize_facts_with_llm().

        Raw facts:

        Christopher Diaz was a white man born in Ryandbury, Rhode Island.
        He worked for Willis Group LLC as a clinical scientist in Lewischester, MI.
        Christopher married Cindy Lopez in 1965.
        Cindy gave him two sons and one daughter.
        Christopher enjoyed being a sports coach for the public school Griffin, where his kids attended.
        He died at the age of 82 in 2014.
        """
        docs: List[str] = [
        "Christopher Diaz was a white man born in Ryandbury, Rhode Island",
        "Christopher Diaz worked for Willis Group LLC as a clinical scientist in Lewischester, MI",
        "Christopher Diaz married Cindy Lopez in 1965",
        "Cindy Lopez gave Christopher Diaz two sons and one daughter",
        "Christopher Diaz enjoyed being a sports coach for the public school Griffin, where Christopher Diaz's kids attended",
        "Christopher Diaz died at the age of 82 in 2014, having been born in approximately 1932 (with a possible one-year ambiguity)"
        ]

        # Build the list of LangChain documents
        documents = []
        for doc in docs:
            documents.append(Document(
                page_content=doc,
                # metadata[0] has high level metadata
                # facts_metadata[i] has metadata for each individual fact.
                metadata={
                    "language": "en - English",                 # 'en - English'
                    "id": str(uuid.uuid4())                     # The unique ID for this fact
                }
                )
            )

        return documents


    def build_langchain_chroma_db(path_chroma_db:Path, documents:List[Document], collection_name:str, verbose:bool=True):
        """
        Adds 'documents' to the local Chroma db 'path_chroma_db' with the collection name 'collection_name'.
        
        """

        # Build a list of the unique ids from doc.metadata['id'] in documents (used later for the call to vector_store.add_documents())
        #uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
        uuids = []
        for doc in documents:
            uuids.append(doc.metadata['id'])
        #print(uuids)        # ['25016fc6-43ac-4260-96a2-174e67fb2b43', 'c00caf02-38d6-409a-89bb-635a574ad180', 'a0a7df19-3416-495c-9e18-fca98a6aab7e', '5a51f198-c4df-428a-90d9-9c518dda48fe', '9d91c213-a333-4a10-9436-38bd8c480a29', '6746f3cc-121c-466c-813c-38b8e8f3c5b9']

        # Initialize the embedding function using the model specified in our config
        embedding_function = OpenAIEmbeddings(
                model=config['embedding_model'],
                #chunk_size=100  # <--- ADD THIS PARAMETER to limit documents per API call.  Lower number causees more independent API calls
            )

        # LangChain & Chroma
        # https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
        # pip install langchain-chroma
        # from langchain_chroma import Chroma

        # Create the vector store
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            # Without 'persist_directory' argument, db is in memory only. 
            persist_directory=str(path_chroma_db)
        )
        if verbose: print(f"Created Chroma vector store for collection '{chroma_collection_name}' in {path_chroma_db}")

        # Add 'documents' and 'uuids' to the vector store.
        vector_store.add_documents(documents=documents, ids=uuids)
        if verbose: print(f"Chroma db vector store updated with {vector_store._collection.count()} embeddings.")


    def sanitize_string_collection_name(input_string):
        """
        Replaces any character in a string that is not alphanumeric, a dash, or an
        underscore with an underscore.

        Args:
            input_string (str): The string to be sanitized.

        Returns:
            str: The sanitized string.
        """
        # The regular expression `[^a-zA-Z0-9_-]` matches any character
        # that is NOT (^) in the specified set of characters.
        # The set includes lowercase letters (a-z), uppercase letters (A-Z),
        # digits (0-9), a hyphen (-), and an underscore (_).
        return re.sub(r'[^a-zA-Z0-9_-]', '_', input_string)


    if doc_format is None: raise Exception(f"doc_format must be either 'LangChain' or List[str]'")
    if not doc_format in ["LangChain", "List[str]"]: raise Exception(f"doc_format must be either 'LangChain' or List[str]'")

    filename_base = "christopher_diaz"

    path_chroma_db = config['path_chroma_db']
    if not isinstance(path_chroma_db, Path): raise Exception("path_chroma_db is not a Path object")

    chroma_collection_name = config['chroma_collection_name']

    # Get the Christopher Diaz set of fake facts. 
    path_file_docs = Path(Path.cwd()).joinpath(f"docs_{filename_base}.pickle")
    path_file_docs_collection_name = Path(Path.cwd()).joinpath(f"docs_{filename_base}_collection_name.pickle")
    
    if rebuild_source_files:
        # Delete folder path_chroma_db
        if path_chroma_db.is_dir(): shutil.rmtree(path_chroma_db)
        # Delete Pickle files 
        if path_file_docs.is_file(): path_file_docs.unlink()
        if path_file_docs_collection_name.is_file(): path_file_docs_collection_name.unlink()
    
    if not path_file_docs.is_file() or not path_file_docs_collection_name.is_file():
        rebuild_source_files = True

        # Get a list of LangChain documents with metadata.
        
        # Sometimes docs_langchain_fake_christopher_diaz() doesn't properly generate a match between len(facts) and len(metadata)
        i = 0
        while True:
            i += 1
            try:
                documents = docs_langchain_fake_christopher_diaz_enriched()
                break
            except Exception as e:
                print(f"{i}\t{e}")
        
        if verbose: print(f"Fetched {len(documents)} enriched documents for '{chroma_collection_name.replace('_',' ')}'")
        #print(f"\n{documents[0].metadata}")        # {'language': 'en - English', 'id': 'e0d3bc36-b85f-4ebf-bbd2-6b466980eae3'}

        # Write the documents to a local pickle file
        with open(path_file_docs, 'wb') as f:
            pickle.dump(documents, f)

        # Write the collection name to a local pickle file
        with open(path_file_docs_collection_name, 'wb') as f:
            pickle.dump(chroma_collection_name, f)

    else:
        # Read the Christopher Diaz documents and collection name
        with open(path_file_docs, 'rb') as f:
            documents = pickle.load(f)
        with open(path_file_docs_collection_name, 'rb') as f:
            stored_collection_name  = pickle.load(f)
        if verbose: print(f"Read {len(documents)} documents and collection name '{stored_collection_name}'")

    if rebuild_source_files:
        # Add 'documents' as 'chroma_collection_name' to a Chroma db persisted to the local file 'path_chroma_db'.
        build_langchain_chroma_db(path_chroma_db, documents, chroma_collection_name, verbose)

    if doc_format == "LangChain": return documents

    # documents are formatted as a LangChain document.  
    # [Document(metadata={'section': 'life', 'source_doc': 'FakeWikiBio', 'language': 'en - English', 'summary': 'Biography of Christopher Diaz', 'id': 'ff1c2674-bc65-440c-b53b-e9f4a52b8f2e'}, page_content='Christopher Diaz was a white man born in Ryandbury, Rhode Island'), Document(metadata={'section': 'employment', 'source_doc': 'FakeWikiBio', 'language': 'en - English', 'summary': 'Biography of Christopher Diaz', 'id': 'ad756cef-de4c-4507-bed4-d710e16c4f1c'}, page_content='Christopher Diaz worked for Willis Group LLC as a clinical scientist in Lewischester, MI'), Document(metadata={'section': 'family', 'source_doc': 'FakeWikiBio', 'language': 'en - English', 'summary': 'Biography of Christopher Diaz', 'id': 'c866bf07-eba8-405a-9248-fdf88a703c30'}, page_content='Christopher Diaz married Cindy Lopez in 1965'), Document(metadata={'section': 'family', 'source_doc': 'FakeWikiBio', 'language': 'en - English', 'summary': 'Biography of Christopher Diaz', 'id': '0e0c9233-9998-4cf2-b46b-412d7dd0f450'}, page_content='Cindy Lopez gave Christopher Diaz two sons and one daughter'), Document(metadata={'section': 'family', 'source_doc': 'FakeWikiBio', 'language': 'en - English', 'summary': 'Biography of Christopher Diaz', 'id': '6feda629-737a-4fd7-92c2-1291992e73d6'}, page_content="Christopher Diaz enjoyed being a sports coach for the public school Griffin, where Christopher Diaz's kids attended"), Document(metadata={'section': 'life', 'source_doc': 'FakeWikiBio', 'language': 'en - English', 'summary': 'Biography of Christopher Diaz', 'id': 'a13dadd1-56c7-4de2-bc88-bd90880b4031'}, page_content='Christopher Diaz died at the age of 82 in 2014, having been born in either 1931 or 1932.')]
    # Format documents as a simple list of str.
    docs = []
    for doc in documents:
        docs.append(doc.page_content)

    return docs


def search_docs(query: str) -> str:
    """
    'vector_search' + 'keyword_search' (BM25)

    USAGE:

    query = "What is the name of the state where the children of Christopher Diaz attended public school?"
    documents = search_docs(query)
    print(documents)

    """

    # Ensemble (Hybrid) Retriever for vector search results AND BM25 (Best Match 25) search results.
    # The Ensemble (Hybrid) Retriever combines the results of the two retrievers  (vector & BM25) 
    # and re-ranks them using Reciprocal Rank Fusion (RRF).
    # Reciprocal Rank Fusion (RRF) is designed to merge and re-rank the results from multiple different retrieval systems or queries.

    # LangChain Ensemble Retriever
    # The latest stable version of LangChain (Python) is v1.1.0.
    # The class EnsembleRetriever (in the langchain.retrievers.ensemble module) appears in documentation indicating version 0.2.15 in the langchain namespace.
    # The legacy retrievers — including EnsembleRetriever — have been moved out of the core langchain namespace and into a separate compatibility package: langchain-classic.
    # pip install -U langchain-classic
    #from langchain_classic.retrievers import EnsembleRetriever
    # All below are obsolete:
    #from langchain_experimental.retrievers import EnsembleRetriever
    #from langchain_community.retrievers import EnsembleRetriever
    #from langchain_core.retrievers import EnsembleRetriever
    #from langchain.retrievers import EnsembleRetriever
    #from langchain.retrievers.ensemble import EnsembleRetriever

    #from langchain_community.retrievers import BM25Retriever

    # BM25 (Best Match 25) Search
    # BM25, a form of sparse retrieval, is likely to be more effective than Chroma's vector search when:
    #   You Need Exact Term Matching
    #   Queries are literal (precise keywords)
    # BM25 excels at finding documents with exact keyword matches and is great for identifying specific, rare terms (like names, codes, or domain-specific jargon).
    # bm25s is a better library.
    # Initialize the BM25 (Sparse) Retriever
    # BM25 is purely keyword-based and is created from the raw text documents.
    documents = get_docs_christopher_diaz(doc_format="LangChain", verbose=False)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5    # Return top 5 results

    # Initialize the embedding function using the model specified in our config
    embedding_function = OpenAIEmbeddings(
            model=config['embedding_model'],
            #chunk_size=100  # <--- ADD THIS PARAMETER to limit documents per API call.  Lower number causees more independent API calls
        )

    # Reconnect to the vector store
    vector_store = Chroma(
        collection_name=config['chroma_collection_name'],
        embedding_function=embedding_function,
        # Without 'persist_directory' argument, db is in memory only. 
        persist_directory=str(config['path_chroma_db'])
    )
    print(f"Connected Chroma db to collection '{config['chroma_collection_name']}' in {config['path_chroma_db']}")

    # Configure the Chroma vector store retriever to return the top 5 results.
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Initialize the Ensemble (Hybrid) Retriever
    # It combines the results of the two retrievers and re-ranks them using 
    # Reciprocal Rank Fusion (RRF), weighted by the 'weights' parameter.
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5] # Equal weighting for both BM25 and Chroma results
    )


    # The EnsembleRetriever invokes BOTH retrievers and combines the results
    retrieved_docs = ensemble_retriever.invoke(query)

    #print(f"search_docs() Vector + BM25 found {len(retrieved_docs)} documents")
    print(f"\nsearch_docs() Vector + BM25 found {len(retrieved_docs)} results:")
    for i,doc in enumerate(retrieved_docs):
        print(f"{i}\t{doc.page_content}")
    """
    """

    # Convert from LangChain to List[str]
    docs = []
    for doc in retrieved_docs:
        docs.append(doc.page_content)

    return docs


# --- Search Web & Utility Functions ---


def search_web(query: str) -> List[str]:
    """
    USAGE:
    
    query = "Who is the CEO of Marando Industries Inc?"
    results = search_web(query)
    print(f"type(results): {type(results)}")
    for i,result in enumerate(results):
        print(f"{i}\t{result}")
    """

    web_search_tool = TavilySearch(k=3)

    console = Console()

    # Invoke the Tavily search tool. The output is a dictionary (results_dict).
    #results_dict: Dict[str, Any] = web_search_tool.invoke({"query": query})
    #results_dict = web_search_tool.run(query)
    results_dict = web_search_tool.invoke({"query": query})

    if not isinstance(results_dict, dict):        
        # Gracefully handle the tool failure by logging and returning an empty list
        console.print(f"[!] Web search failed for query: '{query}'")
        console.print(f"[!] Tool returned an invalid format or error: {results_dict}")
        return [] # <--- THIS IS THE CRITICAL CHANGE: Return [] instead of raising error


    # CRITICAL FIX: Extract the actual list of search results from the 'results' key.
    # The results list is nested inside the dictionary. We use .get() for safety.
    results_list: List[Dict[str, str]] = results_dict.get('results', [])

    #print(results_list)
    # [{'url': 'https://www.datanyze.com/companies/marando-industries/405809135', 'title': 'Marando Industries Company Profile', 'content': "Who is Marando Industries's CEO? Marando Industries's CEO is Richard Marando. What are some additional names or alternative spellings that users use while", 'score': 0.9195344, 'raw_content': None}, {'url': 'https://rocketreach.co/marando-industries-inc-management_b45bd854fc5fd69e', 'title': 'Marando Industries, Inc. Information', 'content': 'Who is the President and CEO of Marando Industries, Inc.? Richard Marando is the President and CEO of Marando Industries, Inc..', 'score': 0.9174287, 'raw_content': None}, {'url': 'https://www.zoominfo.com/c/marando-industries-inc/405809135', 'title': 'Marando Industries - Overview, News & Similar companies', 'content': "Marando Industries's CEO is Richard Marando How do I contact Marando Industries? Marando Industries contact info: Phone number: (610) 621-2536 Website: www.", 'score': 0.8868231, 'raw_content': None}, {'url': 'https://www.zoominfo.com/pic/marando-industries-inc/405809135', 'title': 'Marando Industries: Employee Directory', 'content': "Marando Industries's CEO is Richard Marando Who are Marando Industries key employees? Some of Marando Industries key employees are Carole Wloczewski, Cole", 'score': 0.8827621, 'raw_content': None}, {'url': 'https://www.zoominfo.com/p/Richard-Marando/72770053', 'title': 'Richard Marando, President & Chief Executive Officer at ...', 'content': 'Richard Marando is the President & Chief Executive Officer at Marando Industries based in Reading, Pennsylvania. Previously, Richard was the Director, Advanced', 'score': 0.882276, 'raw_content': None}]
    # A list of dictionaries with keys: url, title, content

    results = []
    for result in results_list:
        results.append(result['content'])

    return results
    
    # Format the results into a list of LangChain Document objects.
    # The loop now iterates over the list of result dictionaries.
    return [
        Document(
            # 'res' is now correctly a dictionary, allowing string-key access
            page_content=res["content"],
            # We store the source URL in the 'metadata' dictionary for citations.
            metadata={"source": res["url"]}
        ) 
        # Iterate over the extracted list
        for res in results_list
    ]

# --- DSPy Configuration ---

openai_key = os.environ['OPENAI_API_KEY']
# Using the modern dspy.LM() factory function
lm = dspy.LM("gpt-4o-mini", api_key=openai_key) 
dspy.configure(lm=lm)


# --- Agent Signatures ---

class MasterPlanSignature(dspy.Signature):
    """
    Given a complex user query, create a step-by-step master plan of search requirements 
    to gather all necessary data to construct the final answer. Each step must be a simple,
    independent, and targeted search query.
    """
    user_query: str = dspy.InputField(desc="The user's original, complex question.")
    master_plan_steps: List[Dict[str, str]] = dspy.OutputField(
        desc="A list of steps. Each step is a dictionary with the single key 'master_plan_query' "
             "containing the search query for that step. Example: [{'master_plan_query': '...'}, {...}]."
    )
    
class SummarizeRetrievalSignature(dspy.Signature):
    """
    Summarize the raw content retrieved from a search (docs or web) to create a concise step history entry.
    """
    master_plan_query: str = dspy.InputField(desc="The question for which the content was retrieved.")
    raw_retrieved_content: str = dspy.InputField(desc="The raw, unsummarized content from the search tool.")
    summary: str = dspy.OutputField(desc="A concise summary of the retrieved content that answers the master_plan_query. If no useful content was retrieved, return the exact string 'NO_USEFUL_INFO_FOUND'.")

class RawContentCheckSignature(dspy.Signature):
    """
    Given the user's original query and a block of newly retrieved raw text,
    determine if the raw text contains the full and sufficient answer to the user query.
    """
    user_query: str = dspy.InputField(desc="The user's original question (e.g., 'What is the maiden name of Cindy Diaz?').")
    raw_retrieved_content: str = dspy.InputField(desc="The raw, unsummarized text content from a search tool.")
    
    reflection_output: str = dspy.OutputField(
        desc="Return 'ANSWER_FOUND: [The direct answer to the user_query]' if the content is sufficient. "
             "Otherwise, return 'ANSWER_NOT_FOUND'."
    )

class ReflectSignature(dspy.Signature):
    """
    Analyze the step history against the master plan query to determine if the question has been sufficiently answered.
    """
    master_plan_query: str = dspy.InputField(desc="The specific question for the current step.")
    step_history: List[Dict[str, str]] = dspy.InputField(desc="The chronological history of all search attempts and summaries for this step.")
    is_answer_sufficient: bool = dspy.OutputField(desc="True if the step_history contains a clear, factual answer to the master_plan_query, False otherwise.")

class FinalizeStepSignature(dspy.Signature):
    """Summarize the complete step history to provide a definitive answer for the current step query."""
    master_plan_query: str = dspy.InputField(desc="The specific question for the current step.")
    step_history: List[Dict[str, str]] = dspy.InputField(desc="The complete history of all search attempts and summaries for this step.")
    step_summary: str = dspy.OutputField(desc="A final, factual summary that directly answers the master_plan_query based on the step_history. If no answer could be found in the entire history, state that fact clearly.")

class FinalizeMasterPlanSignature(dspy.Signature):
    """
    Review the final summaries from all steps in the master plan and synthesize a final, coherent answer 
    to the user's original query.
    """
    user_query: str = dspy.InputField(desc="The user's original question.")
    step_summaries: List[str] = dspy.InputField(desc="A list of final summaries, one for each step of the master plan.")
    final_answer: str = dspy.OutputField(desc="A single, comprehensive answer to the user's query.")

# --- Agent Modules (Revised) ---

class MasterPlanAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.planner = dspy.ChainOfThought(MasterPlanSignature)

    def forward(self, user_query):
        result = self.planner(user_query=user_query)
        return result.master_plan_steps

class SearchDocsAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        # SearchDocsAgent only needs to run the summarizer now, the main search is handled procedurally
        self.summarizer = dspy.ChainOfThought(SummarizeRetrievalSignature)

    def forward(self, master_plan_query, raw_content):
        # This forward call is now simplified to only summarize the raw content
        if not raw_content:
            return "NO_USEFUL_INFO_FOUND"
            
        summary = self.summarizer(
            master_plan_query=master_plan_query,
            raw_retrieved_content=raw_content
        ).summary
        
        return summary

class SearchWebAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarizer = dspy.ChainOfThought(SummarizeRetrievalSignature)

    def forward(self, master_plan_query, raw_content):
        # This forward call is now simplified to only summarize the raw content
        if not raw_content:
            return "NO_USEFUL_INFO_FOUND"

        summary = self.summarizer(
            master_plan_query=master_plan_query,
            raw_retrieved_content=raw_content
        ).summary
        
        return summary

class RawContentCheckAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.checker = dspy.Predict(RawContentCheckSignature)

    def forward(self, user_query, raw_content):
        prediction = self.checker(
            user_query=user_query,
            raw_retrieved_content=raw_content
        )
        return prediction.reflection_output
        
class ReflectAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.reflector = dspy.Predict(ReflectSignature)

    def forward(self, master_plan_query, step_history):
        history_str = json.dumps(step_history, indent=2)
        
        prediction = self.reflector(
            master_plan_query=master_plan_query,
            step_history=history_str
        )
        return prediction.is_answer_sufficient

class FinalizeStepAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.finalizer = dspy.Predict(FinalizeStepSignature)

    def forward(self, master_plan_query, step_history):
        history_str = json.dumps(step_history, indent=2)
        prediction = self.finalizer(
            master_plan_query=master_plan_query,
            step_history=history_str
        )
        return prediction.step_summary

class FinalizeMasterPlanAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.finalizer = dspy.Predict(FinalizeMasterPlanSignature)

    def forward(self, user_query, step_summaries):
        prediction = self.finalizer(
            user_query=user_query,
            step_summaries=step_summaries
        )
        return prediction.final_answer


# --- Pipeline Module (Revised) ---

class MultiAgentPipeline(dspy.Module):
    def __init__(self, max_step_iterations: int = 3):
        super().__init__()
        
        self.max_step_iterations = max_step_iterations

        self.master_planner = MasterPlanAgent()
        self.raw_content_checker = RawContentCheckAgent() # NEW
        self.search_docs_agent = SearchDocsAgent()
        self.search_web_agent = SearchWebAgent()
        self.reflector = ReflectAgent()
        self.step_finalizer = FinalizeStepAgent()
        self.master_finalizer = FinalizeMasterPlanAgent()

    def forward(self, user_query):
        
        console = Console()
        console.rule(f"[bold magenta]Starting Pipeline for Query: '{user_query}'[/bold magenta]")
        
        # 1. Master Plan Creation
        master_plan_steps_raw = self.master_planner(user_query=user_query)
        master_plan_queries = [step['master_plan_query'] for step in master_plan_steps_raw]
        
        console.print(f"\n[bold yellow]Master Plan ({len(master_plan_queries)} Steps):[/bold yellow]")
        for i, query in enumerate(master_plan_queries):
            console.print(f"  Step {i+1}: {query}")
        console.rule()

        all_step_summaries = []
        pipeline_completed = False 

        # 2. Step-by-Step Execution
        for step_index, master_plan_query in enumerate(master_plan_queries):
            
            if pipeline_completed:
                break
                
            step_number = step_index + 1
            console.rule(f"[bold cyan]-- Step {step_number}/{len(master_plan_queries)}: {master_plan_query} --[/bold cyan]")
            
            step_history = []
            is_answered = False
            last_docs_result_was_useful = True 
            
            for iteration in range(1, self.max_step_iterations + 1):
                
                if pipeline_completed:
                    break
                    
                console.print(f"\n[bold blue]  Iteration {iteration}:[/bold blue]")

                # A. Tool Selection Logic (Local-First, Web-Fallback)
                if iteration == 1:
                    recommended_tool = 'search_docs'
                elif iteration > 1 and not last_docs_result_was_useful:
                    recommended_tool = 'search_web'
                else:
                    # If last search was useful, break to check if step is answered by ReflectAgent
                    break 

                console.print(f"    -> [green]Tool Selector:[/green] Recommended tool: {recommended_tool}")

                # B. Execute Search and Get Raw Content
                raw_content = ""
                if recommended_tool == 'search_docs':
                    raw_content = search_docs(master_plan_query)
                elif recommended_tool == 'search_web':
                    raw_content = search_web(master_plan_query)

                # B.1. IMMEDIATE RAW CONTENT CHECK (NEW LOGIC)
                if raw_content:
                    check_result = self.raw_content_checker(user_query=user_query, raw_content=raw_content)
                    
                    if check_result.startswith('ANSWER_FOUND:'):
                        direct_answer = check_result.replace('ANSWER_FOUND: ', '').strip()
                        
                        # Use the direct answer as the summary for this final, successful step
                        summary = f"Direct answer to the overall user query found: {direct_answer}"
                        pipeline_completed = True
                        
                        console.print(f"    -> [red]Pipeline Reflection (Full Query):[/red] ANSWER FOUND! Terminating early.")
                        console.print(f"    -> [red]Answer Snippet:[/red] {direct_answer[:100]}...")

                    else: # ANSWER_NOT_FOUND, proceed with step-level summarization
                        
                        # B.2. Step-level Summarization
                        if recommended_tool == 'search_docs':
                            summary = self.search_docs_agent(master_plan_query=master_plan_query, raw_content=raw_content)
                        elif recommended_tool == 'search_web':
                            summary = self.search_web_agent(master_plan_query=master_plan_query, raw_content=raw_content)
                        
                        
                        # Update utility flag for next iteration
                        if recommended_tool == 'search_docs':
                            if summary == 'NO_USEFUL_INFO_FOUND':
                                last_docs_result_was_useful = False
                            else:
                                last_docs_result_was_useful = True
                        elif recommended_tool == 'search_web':
                             # Web search is the final attempt, so we reset the docs flag.
                             last_docs_result_was_useful = False 
                        
                        # Console output
                        if summary == 'NO_USEFUL_INFO_FOUND':
                            console.print("    -> [green]Search Result Summary:[/green] No useful information found.")
                        else:
                            console.print(f"    -> [green]Search Result Summary:[/green] {summary[:100]}...")

                        # D. Reflect and Check for Step Completion (original logic)
                        if summary != 'NO_USEFUL_INFO_FOUND' and not pipeline_completed:
                            is_answered = self.reflector(
                                master_plan_query=master_plan_query, 
                                step_history=step_history
                            )
                            
                            console.print(f"    -> [green]Step Reflection (Step Query):[/green] Answer sufficiently found: {is_answered}")

                            if is_answered:
                                break # breaks iteration loop if step is answered

                else: # No raw content retrieved at all
                    summary = "NO_USEFUL_INFO_FOUND"
                    last_docs_result_was_useful = False # Treat lack of content as not useful

                
                # C. Update History
                history_entry = {
                    "iteration": iteration,
                    "tool_used": recommended_tool,
                    "summary": summary
                }
                step_history.append(history_entry)
                
                if pipeline_completed or iteration >= self.max_step_iterations:
                    break
            
            # E. Finalize Step
            if not pipeline_completed:
                final_step_summary = self.step_finalizer(
                    master_plan_query=master_plan_query, 
                    step_history=step_history
                )
            else:
                # If pipeline completed, the last summary is the definitive answer for the overall query.
                final_step_summary = summary
                # Ensure the history includes the successful final entry
                if not any('Direct answer to the overall user query found' in h['summary'] for h in step_history):
                     step_history.append(history_entry)
            
            all_step_summaries.append(final_step_summary)
            
            console.print(f"\n[bold magenta]Step {step_number} Final Summary:[/bold magenta] {final_step_summary}")
            console.rule()

            if pipeline_completed:
                break # breaks step loop
        
        # 3. Final Answer Synthesis
        console.rule("[bold yellow]Synthesizing Final Answer[/bold yellow]")
        final_answer = self.master_finalizer(
            user_query=user_query,
            step_summaries=all_step_summaries
        )
        
        console.rule("[bold green]Final Answer[/bold green]")
        console.print(final_answer)
        console.rule()

        return final_answer




if __name__ == '__main__':
    pass

    # --- Define the user query ---

    # Example Multi-Hop Query: requires multiple queries against the local documents to answer the question. 
    user_query = "What is the name of the state where the children of Christopher Diaz attended public school?"
    #user_query = "What is the maiden name of Cindy Diaz?"

    #user_query = "What public school did Christopher Dias attend?"
    #user_query = "Was Christopher Diaz employed as a sports coach for Willis Group LLC?"
    #user_query = "Was Christopher Diaz's first child born in the year 1930?"

    # Example web queries
    #user_query = "Who is the CEO of Marando Industries Inc?"
    #user_query = "How can I access DeepSeek OCR from a laptop without a GPU?"

    # --- Initialize and Run the Pipeline ---
    pipeline = MultiAgentPipeline(max_step_iterations=config['max_step_iterations'])
    
    final_response = pipeline(user_query=user_query)

    print("\n\n--- Final Result ---")
    print(f"User Query: {user_query}")
    print(f"Answer: {final_response}")
    print("--------------------")

