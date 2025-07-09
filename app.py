import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Title
st.title("📄 Document Analyzer using LLMs")

# 1. Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    # 2. Save PDF temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    # 3. Load PDF
    st.info("📥 Extracting text from PDF...")
    loader = PyMuPDFLoader("temp.pdf")
    documents = loader.load()
    
    # 4. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    st.success(f"✅ Split into {len(chunks)} chunks")

    # 5. Create Embeddings
    st.info("🔄 Generating embeddings...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")  # make sure it's downloaded via `ollama pull`
    
    # 6. Vector DB - FAISS
    db = FAISS.from_documents(chunks, embeddings)
    
    # 7. Initialize LLM
    llm = Ollama(model="llama3")  # Run `ollama run llama3` first

    # 8. Create Retrieval Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=True
    )

    # 9. User Input
    query = st.text_input("💬 Ask something about the PDF:")

    if query:
        st.info("🤖 Thinking...")
        result = qa_chain.invoke(query)
        
        # 10. Show answer
        st.subheader("🧠 Answer:")
        st.write(result["result"])
        
        # Optional: Show sources
        with st.expander("📚 Source Chunks"):
            for doc in result["source_documents"]:
                st.markdown(doc.page_content)
