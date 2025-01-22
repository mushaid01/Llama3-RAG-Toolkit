# Llama3-RAG-Toolkit 🚀🤖

Welcome to the **Llama3-RAG-Toolkit**! This repository showcases an advanced Retrieval-Augmented Generation (RAG) pipeline using **Meta Llama 3 (8B-Instruct)** for highly accurate and contextually aware question-answering systems. 🌟

---

## ✨ Key Features

- **State-of-the-Art RAG Workflow**: Combines powerful retrieval mechanisms with Meta's Llama 3 model for dynamic and precise responses. 📚
- **Highly Optimized**: Utilizes **Bits and Bytes (bnb)** quantization for efficient 4-bit precision, making it suitable for GPU deployments. ⚡
- **Customizable Pipelines**: Flexible configurations for document retrieval, text generation, and embeddings. 💻
- **Multimodal Document Support**: Easily integrates with PDF loaders and supports chunked document retrieval. 📄
- **Interactive Testing**: Ready-to-use functions for testing RAG pipelines and visualizing results. ✅

---

## 🛠️ Technical Stack

- **Language Model**: Meta Llama 3 (8B-Instruct) 🚀
- **Vector Database**: Chroma for efficient document storage and retrieval.
- **Embeddings**: SentenceTransformers for semantic representation.
- **Frameworks**: LangChain for pipeline integration and orchestration.
- **Development Environment**: Python and Jupyter Notebooks for interactive exploration.

---

## 📋 Installation Guide

Follow these steps to set up the project:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/mushaid01/Llama3-RAG-Toolkit.git
   cd Llama3-RAG-Toolkit
   ```

2. **Install Dependencies**:
   ```bash
   pip install transformers==4.33.0 accelerate==0.22.0 einops==0.6.1 langchain==0.0.300 xformers==0.0.21 \
   bitsandbytes==0.41.1 sentence_transformers==2.2.2 chromadb==0.4.12
   ```

3. **Set Up Environment Variables**:
   - Add your Hugging Face token and any required API keys to a `.env` file:
     ```
     HUGGINGFACE_TOKEN=your_huggingface_token
     ```

4. **Prepare the Model and Tokenizer**:
   - Load the Meta Llama 3 model with 4-bit quantization:
     ```python
     from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

     bnb_config = BitsAndBytesConfig(
         load_in_4bit=True,
         bnb_4bit_quant_type='nf4',
         bnb_4bit_use_double_quant=True
     )

     model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", 
                                                  quantization_config=bnb_config)
     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
     ```

5. **Run the Notebook**:
   - Open the provided Jupyter notebook and execute the cells to load your data and test the pipeline.

---

## ⚙️ Usage Instructions

1. **Test the Model**:
   Use the pre-built `test_model` function to evaluate the Llama 3 pipeline with a sample query:
   ```python
   response = test_model(tokenizer, query_pipeline, "What are the future prospects of AI in finance?")
   print(response)
   ```

2. **Load and Split Documents**:
   Load your PDFs or other document sources and split them into manageable chunks:
   ```python
   from langchain.document_loaders import PyPDFLoader
   from langchain.text_splitter import RecursiveCharacterTextSplitter

   loader = PyPDFLoader("example.pdf")
   documents = loader.load()
   splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
   splits = splitter.split_documents(documents)
   ```

3. **Build the RAG Pipeline**:
   Create a retriever and integrate it into a retrieval-augmented QA chain:
   ```python
   from langchain.vectorstores import Chroma
   from langchain.embeddings import HuggingFaceEmbeddings
   from langchain.chains import RetrievalQA

   embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
   vectordb = Chroma.from_documents(splits, embedding=embeddings, persist_directory="chroma_db")
   retriever = vectordb.as_retriever(search_kwargs={"score_threshold": 0.01, "k": 8})

   qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, verbose=True)
   ```

4. **Test the RAG Pipeline**:
   ```python
   query = "What are the global future prospects of the banking industry?"
   test_rag(qa, query)
   ```

---

## 🌍 Applications

- **AI-Powered Knowledge Systems**: Build advanced Q&A systems for enterprises.
- **Document Summarization**: Extract meaningful insights from large document collections.
- **Research Assistance**: Aid researchers with precise and context-aware answers.
- **Industry-Specific Solutions**: Apply RAG pipelines to finance, healthcare, and more.

---

## 💬 Contributing

We welcome contributions to enhance this project! Feel free to submit pull requests, open issues, or suggest features.


---

Made with ❤️ and AI by [Mir Mushaidul Islam].
