from flask import Flask, request, jsonify
import asyncio
import os
import uuid
from langchain import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from flask_cors import CORS
import logging
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import numpy as np
from PIL import Image

# Set the environment variable to handle the OpenMP runtime issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)
CORS(app, origins="*")

logging.basicConfig(level=logging.INFO)

poppler_path = r"C:\poppler-24.07.0\Library\bin"
local_llm = r"D:\BDO\Chat-Bot\New folder (2)\zephyr-7b-alpha.Q4_K_M.gguf"

config = {
    "max_new_tokens": 2048,
    "context_length": 4096,
    "repetition_penalty": 1.4,
    "temperature": 0.2,
    "top_k": 50,
    "top_p": 0.7,
    "stream": True,
    "threads": int(os.cpu_count() / 2),
}

llm_init = CTransformers(
    model=local_llm,
    model_type="mistral",
    gpu_layers=50,
    config=config,
    lib="avx2",
    n_ctx=2048,
    **config
)

custom_prompt_template = """Use only the context mentioned below to answer the user's question.
If you don't know the answer, just say that you don't know.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

ocr = PaddleOCR(use_angle_cls=True, lang='en')

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def extract_text_from_pdf(file_path):
    images = convert_from_path(file_path, poppler_path=poppler_path)
    text = ""
    for image in images:
        image_np = np.array(image)  # Convert image to NumPy array
        result = ocr.ocr(image_np, cls=True)
        for line in result:
            text += ''.join([word_info[1][0] for word_info in line])
            text += '\n'
    return text

def process_documents(file_path):
    documents = []
    text = extract_text_from_pdf(file_path)
    if text.strip():
        doc = Document(
            page_content=text,
            metadata={"original_pdf_name": os.path.basename(file_path)}
        )
        documents.append(doc)
    else:
        logging.error(f"Failed to extract any text from {file_path}")
    
    return documents

def make_embedder():
    model_name = "all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    return HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

async def qa_bot(db):
    prompt = PromptTemplate(
        input_variables=["context", "question", "history"],
        template=custom_prompt_template,
    )
    memory = ConversationBufferMemory(memory_key="history", input_key="question")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_init,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
    return qa_chain

def generate_unique_directory(filename):
    # Sanitize filename to use in directory name (remove special characters, etc.)
    safe_filename = ''.join(char for char in filename if char.isalnum() or char in ('_', '-'))
    
    unique_dir = os.path.join("./upload", safe_filename)
    
    # Check if the directory already exists
    if os.path.exists(unique_dir):
        return unique_dir, True  # Return True if the directory already exists
    
    os.makedirs(unique_dir, exist_ok=True)
    print("unique_dir", unique_dir)
    return unique_dir, False  # Return False if a new directory was created


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    print("file",file)
    print("filename",file.name)
    file_path = os.path.join("uploads", file.filename)
    print("file_path",file_path)
    file.save(file_path)

    # Process documents
    documents = process_documents(file_path)
   
    logging.info(f"Processed documents: {documents}")

    if not documents:
        return jsonify({"error": "No documents could be processed from the file"}), 400

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=20, separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    print(texts)
    logging.info(f"Split texts: {texts}")

    if not texts:
        return jsonify({"error": "No texts could be split from the documents"}), 400

     # Create a unique directory for the new upload
    unique_persist_directory, already_exists = generate_unique_directory(file.filename)

    if already_exists:
        message = "Directory already exists. Reusing existing embeddings."
    else:
        message = "File processed and QA chain initialized."
    # Load embedding model
    hf = make_embedder()

    # Create Chroma DB with the unique directory
    db = Chroma.from_documents(
        documents=texts, embedding=hf, persist_directory=unique_persist_directory
    )

    # Initialize QA chain with the new database
    chain = asyncio.run(qa_bot(db))
    app.config["qa_chain"] = chain
    app.config["persist_directory"] = unique_persist_directory  # Save the current directory

    return jsonify({"message": message}), 200

@app.route("/ask", methods=["POST"])
def ask_question():
    chain = app.config.get("qa_chain")
    if not chain:
        return jsonify({"error": "QA chain not initialized"}), 400
    question = request.form.get("query")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    res = chain({"query": question})
    answer = res["result"]
    return jsonify({"answer": answer})

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(host="0.0.0.0", port=5000)
