import os
import io
import uuid
import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.docstore.document import Document

app = Flask(__name__)
CORS(app)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

print("--- Initializing Exam Prep RAG Backend... ---")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0.5)

session_retrievers = {}

def format_docs(docs):
    return "\n\n".join(f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}" for doc in docs)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    files = request.files.getlist('file')
    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400

    all_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    processed_filenames = []

    for file in files:
        filename = file.filename
        if filename.endswith('.pdf'):
            pdf_reader = PdfReader(io.BytesIO(file.read()))
            for i, page in enumerate(pdf_reader.pages):
                page_content = page.extract_text() or ""
                doc = Document(page_content=page_content, metadata={"source": f"{filename} - Page {i+1}"})
                all_documents.append(doc)
            processed_filenames.append(filename)
        elif filename.endswith('.txt'):
            content = file.read().decode('utf-8')
            doc = Document(page_content=content, metadata={"source": filename})
            all_documents.append(doc)
            processed_filenames.append(filename)
    
    if not all_documents:
        return jsonify({"error": "No processable files found or files are empty."}), 400

    chunked_documents = text_splitter.split_documents(all_documents)
    
    session_id = str(uuid.uuid4())
    
    faiss_vector_store = FAISS.from_documents(chunked_documents, embeddings)
    faiss_retriever = faiss_vector_store.as_retriever(search_kwargs={'k': 5})
    
    bm25_retriever = BM25Retriever.from_documents(chunked_documents)
    bm25_retriever.k = 5
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )
    
    session_retrievers[session_id] = ensemble_retriever
    
    print(f"--- Created new session: {session_id} with {len(chunked_documents)} chunks from {len(processed_filenames)} file(s) ---")
    
    return jsonify({"session_id": session_id, "message": f"Successfully processed {len(processed_filenames)} file(s)."}), 200

@app.route('/ask', methods=['GET'])
def ask_question():
    question = request.args.get('question', '').lower().strip()
    session_id = request.args.get('session_id')
    chat_history_str = request.args.get('chat_history', '[]')
    chat_history = json.loads(chat_history_str)

    def stream_response():
        small_talk_responses = {
            "hello": "Hello! How can I help you with your document today?",
            "hi": "Hi there! Ready to dive into your study materials?",
            "hey": "Hey! What questions do you have about your document?",
            "thanks": "You're welcome! Is there anything else I can help you with?",
            "thank you": "You're very welcome! Let me know if you need anything else.",
            "bye": "Goodbye! Happy studying!",
            "goodbye": "Goodbye! Feel free to start a new session anytime."
        }
        if question in small_talk_responses:
            yield f"data: {json.dumps({'type': 'answer', 'content': small_talk_responses[question]})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            return

        if not all([question, session_id]):
            yield f"data: {json.dumps({'type': 'error', 'content': 'Missing question or session_id'})}\n\n"
            return

        retriever = session_retrievers.get(session_id)
        if not retriever:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Invalid session_id or session has expired'})}\n\n"
            return

        try:
            if chat_history:
                history_str = "\n".join([f"Human: {h['human']}\nAI: {h['ai']}" for h in chat_history])
                condense_prompt = PromptTemplate.from_template("Rephrase the follow up question to be a standalone question. Chat History: {chat_history}\nFollow Up Input: {question}\nStandalone question:")
                condense_chain = condense_prompt | llm
                standalone_question = condense_chain.invoke({"chat_history": history_str, "question": question}).content
            else:
                standalone_question = question
            
            retrieved_docs = retriever.invoke(standalone_question)
            formatted_context = format_docs(retrieved_docs)
            
            answer_prompt = PromptTemplate.from_template("You are an expert exam preparation assistant. Answer the user's question based only on the provided context. Be detailed and clear. For each piece of information, cite the source document it came from using the format [Source: Document Name - Page X].\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:")
            answer_chain = answer_prompt | llm
            
            for chunk in answer_chain.stream({"context": formatted_context, "question": standalone_question}):
                yield f"data: {json.dumps({'type': 'answer', 'content': chunk.content})}\n\n"
            
            sources = list({doc.metadata["source"]: doc.page_content for doc in retrieved_docs}.items())
            sources_for_frontend = [{"source": src, "content": content} for src, content in sources]
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_for_frontend})}\n\n"

        except Exception as e:
            print(f"An error occurred during /ask stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': 'An internal error occurred'})}\n\n"

    return Response(stream_response(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
