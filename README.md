# **Multi-Agent DSPy RAG and Web Search Pipeline**

This repository contains a Python script that implements a sophisticated, agentic pipeline for answering complex user queries. It strategically combines information retrieved from a local document database (using a hybrid RAG system) and the external web (via the Tavily search tool).

The core goal of the pipeline is to break down a difficult, multi-hop question into manageable steps, execute the necessary searches (local or web) for each step, and synthesize the findings into a single, cohesive final answer.

## **1\. Summary of the Script (api\_lllm\_dspy\_agentic\_pipeline\_v1.0.2.py)**

The script utilizes the DSPy framework to orchestrate a team of specialized agents, ensuring efficient and accurate information retrieval.

### **How the Pipeline Works**

1. **Decomposition:** The initial complex user\_query is fed to the **MasterPlanAgent**, which breaks it down into a list of smaller, targeted master\_plan\_query steps.  
2. **Iterative Search:** The pipeline iterates through these steps. For each step, it attempts to find an answer by first querying the local document database, and then (if necessary) querying the external web.  
3. **Local RAG System:** The local document search uses a **Hybrid Retriever** (LangChain's EnsembleRetriever), which combines:  
   * **Vector Search (Chroma):** Semantic search based on embeddings (text-embedding-3-small).  
   * **Keyword Search (BM25):** Sparse retrieval for exact term matching.  
   * The combined results are re-ranked using Reciprocal Rank Fusion (RRF).  
4. **Web Search:** The web search utilizes the **TavilySearch** tool.  
5. **Early Termination:** Crucially, the **RawContentCheckAgent** immediately analyzes the *raw, unsummarized* search results against the *original user query*. If the complete answer is found at any point, the entire pipeline terminates early, maximizing efficiency and minimizing cost.  
6. **Refinement and Synthesis:** If early termination doesn't occur, the **ReflectAgent** determines if a step is complete. Once all steps are complete, the **FinalizeMasterPlanAgent** takes all partial summaries and synthesizes them into the final response.

### **Core Agents**

| DSPy Agent Module | Function |
| :---- | :---- |
| MasterPlanAgent | Decomposes the complex user query into a series of actionable steps. |
| SearchDocsAgent | Executes the hybrid RAG search against local documents (Chroma \+ BM25) and summarizes the findings for a specific step. |
| SearchWebAgent | Executes the web search using the Tavily tool and summarizes the findings for a specific step. |
| RawContentCheckAgent | Checks the raw search results (local or web) against the **original user query** to enable early pipeline termination. |
| ReflectAgent | Evaluates if a given step's search history is sufficient to answer that step's query. |
| FinalizeStepAgent | Synthesizes a final, factual summary for one completed step. |
| FinalizeMasterPlanAgent | Synthesizes all individual step summaries into the single, final answer for the user's overall query. |

## **2\. Configuration and Setup**

### **Prerequisites**

You need access to two external services:

1. **OpenAI:** For the Large Language Model (LLM) and for the embedding model (text-embedding-3-small).  
2. **Tavily:** For fast, relevant web search grounding.

### **Step 1: Install Dependencies**

The script relies on several Python libraries, specifically pinned to the versions you provided for maximum compatibility.

Create a virtual environment (recommended) and install the packages:

\# Create a virtual environment (optional but recommended)  
python \-m venv venv  
\# Activate the environment (Linux/macOS)  
source venv/bin/activate  
\# Activate the environment (Windows)  
\# .\\\\venv\\\\Scripts\\\\activate

\# Install the required packages  
pip install dspy==3.0.4 \\  
            langchain==1.1.0 \\  
            langchain-chroma==1.0.0 \\  
            langchain-classic==1.0.0 \\  
            langchain-community==0.4.1 \\  
            langchain-core==1.1.0 \\  
            langchain-openai==1.1.0 \\  
            langchain-tavily==0.2.13 \\  
            openai==2.8.1 \\  
            rank-bm25==0.2.2 \\  
            python-dotenv \\  
            rich

### **Step 2: Configure API Keys**

The script uses environment variables, loaded from a .env file, to securely manage API keys.

1. Create a file named **.env** in the root directory of the project (next to the Python script).  
2. Add the following lines to your .env file, replacing the placeholder values with your actual keys:  
   \# Get your key from OpenAI  
   OPENAI\_API\_KEY="your\_openai\_api\_key\_here"

   \# Get your key from Tavily  
   TAVILY\_API\_KEY="your\_tavily\_api\_key\_here"

### **Step 3: Running the Script**

You can now execute the Python pipeline. The script includes a placeholder local document set (a fake biography of Christopher Diaz) and example queries that demonstrate both local RAG and web search capabilities.

python api\_lllm\_dspy\_agentic\_pipeline\_v1.0.2.py

The script will print the execution trace, including the steps generated by the MasterPlanAgent, the search results from the local RAG and Tavily, and the final synthesized answer.