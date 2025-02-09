import faiss
import shutil
import uuid

from flask import Flask, request, jsonify
from omegaconf import OmegaConf
from pathlib import Path

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import ChatOllama


CONF = OmegaConf.load("server_config.yaml")

app = Flask(__name__)
embeddings = OllamaEmbeddings(model=CONF.embedding_model.name, base_url=CONF.ollama.base_url)
chunk_generator = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256, separators=[" ", ",", "\n"]) 
llm_model = ChatOllama(model=CONF.llm_model.name, base_url=CONF.ollama.base_url, temperature=0.0)

rag_prompt = ChatPromptTemplate.from_template("""
    You are an AI assistant that answers questions strictly based on the context provided. Follow these instructions carefully:
        Use only the context provided below to answer the question. Do not use any internal or prior knowledge.
        If the context does not contain enough information to answer the question, respond with: "I don't have enough information to answer that question based on the provided context."
        Do not make up or guess answers. If the context is unclear or insufficient, always return the standard response mentioned above.
        
    Context:
    {context}
    
    Question:
    {question}

    Answer:
    [The model will generate the answer here based on the context or return the standard response if the context is insufficient.]
""")


store = {}
def get_session_history(session_id):
    print(session_id)
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(llm_model | StrOutputParser(), get_session_history)


def load_alread_saved_vector():
    vector_store = {}
    
    pregenerated_vector_store = list(Path("faiss_data/").glob("*"))
    if len(pregenerated_vector_store) != 0:
        for file in pregenerated_vector_store:
            db = FAISS.load_local(folder_path=f"faiss_data/{file.stem}", index_name=f"{file.stem}_index",
                                  embeddings=embeddings, allow_dangerous_deserialization=True, normalize_L2=True)
            vector_store[file.stem] = db

    return vector_store


def create_vector_store():
    return FAISS(
        embedding_function=embeddings,
        index=faiss.IndexFlatIP(CONF.embedding_model.dimensions),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=True,
    )
    
save_index = lambda store, context: store.save_local(folder_path=f"faiss_data/{context}", index_name=f"{context}_index")
format_docs = lambda docs: "\n\n".join([doc.page_content for doc in docs])

def validate_params(data, expected_params):
    for param in expected_params:
        if param not in data:
            return True
    return False


vector_store = load_alread_saved_vector()


@app.route("/rag/query", methods=["POST"])
def query():
    needed_params = ["context", "question"]
    form_data = request.form

    if validate_params(form_data, needed_params):
        return jsonify({"error": f"Missing required '{', '.join(needed_params)}' parameters", "code": 4000}), 400
    if len(vector_store) == 0:
        return jsonify({"error": "No pregenerated vector store found. Generate one before querying.", "code": 4001}), 400

    if form_data.get('context') not in vector_store:
        return jsonify({"error": "No vector store found with the given context", "code": 4002}), 400

    retriever = vector_store[form_data.get('context')].as_retriever(search_type="similarity_score_threshold",
                                                                    search_kwargs = {'k': 3, 'score_threshold': 0.1})
    
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | llm_model
            | StrOutputParser()
        )
    
    llm_response = rag_chain.invoke(form_data.get('question'))

    return jsonify({"status": "ok", "response": llm_response, "code": 200}), 200


# this can be used for code suggestion or any other use case where you don't need a context
# and you just want to query the model directly without any history
@app.route("/llm/query", methods=["POST"])
def query_llm():
    needed_params = ["question"]
    form_data = request.form

    if validate_params(form_data, needed_params):
        return jsonify({"error": f"Missing required '{', '.join(needed_params)}' parameters", "code": 4000}), 400

    rag_chain = RunnablePassthrough() | llm_model | StrOutputParser()
    
    llm_response = rag_chain.invoke(form_data.get('question'))

    return jsonify({"status": "ok", "response": llm_response, "code": 200}), 200


@app.route("/rag/add_document", methods=["POST"])
def add_document():
    needed_params = ["context", "file"]
    form_data = request.form.to_dict()
    form_data.update(request.files.to_dict())

    if validate_params(form_data, needed_params):
        return jsonify({"error": f"Missing required '{', '.join(needed_params)}' parameters", "code": 4000}), 400

    file = request.files["file"]
    if file.filename == "" or not file.filename.endswith(".txt"):
        return jsonify({"error": "No file selected or other file format provided", "code": 4000}), 400
        
    file_content = file.read().decode("utf-8")
    if form_data.get('context') not in vector_store:
        vector_store[form_data.get('context')] = create_vector_store()
    
    chunks = [Document(page_content=chunk) for chunk in chunk_generator.split_text(file_content)]
    vector_store[form_data.get('context')].add_documents(chunks)

    save_index(vector_store[form_data.get('context')], form_data.get('context'))

    return jsonify({"status": "success", "code": 200}), 200


@app.route("/rag/delete", methods=["DELETE"])
def delete_vector_store():
    needed_params = ["context"]
    form_data = request.form.to_dict()
    
    if validate_params(form_data, needed_params):
        return (jsonify({"error": f"Missing required '{', '.join(needed_params)}' parameters", "code": 4000}), 400,)
    
    if form_data.get('context') not in vector_store:
        return jsonify({"error": "No vector store found with the given context", "code": 4002}), 400

    if form_data.get('context') in vector_store:
        del vector_store[form_data.get('context')]
        shutil.rmtree(f"faiss_data/{form_data.get('context')}")
    
    return jsonify({"status": "success", "code": 200}), 200


@app.route("/llm/chat", methods=["POST"])
def chat():
    needed_params = ["question"]
    form_data = request.form

    if validate_params(form_data, needed_params):
        return jsonify({"error": f"Missing required '{', '.join(needed_params)}' parameters", "code": 4000}), 400

    if "session_id" not in form_data:
        session_id = str(uuid.uuid4())
    else:
        session_id = form_data.get('session_id')
    
    response = with_message_history.invoke([HumanMessage(content=form_data.get('question'))],
                                           config={"session_id": session_id})

    return jsonify({"status": "ok", "response": response, "session_id": session_id, "code": 200}), 200


@app.route("/rag/__health_check", methods=["GET"])
def __health_check():
    return jsonify({"status": "ok", "code": 200}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8800, debug=False)



"""
handle mutliple file formats
    Pinecone - for replacement of faiss
    chromadb - for replacement of faiss
""" 