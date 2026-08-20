import os
from typing import TypedDict, Annotated

import streamlit as st
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="centered",
)

# ============================================================
# Step-1 Building the RAG retrieval
# ============================================================

@st.cache_resource(show_spinner="Loading knowledge base...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource(show_spinner="Indexing PDFs...")
def build_retriver(pdf_path: str):
    embeddings = get_embeddings()
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(document)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


@st.cache_resource(show_spinner="Starting up assistant...")
def build_app():
    academic_retriever = build_retriver("academics_handbook.pdf")
    fee_retriever = build_retriver("fee_structure.pdf")
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.4)

    # ============================================================
    # Step-2 State
    # ============================================================
    class State(TypedDict):
        programme: str
        messages: Annotated[list, add_messages]
        query_type: str
        retrieved_context: str

    # ============================================================
    # Step-3 Nodes generation
    # ============================================================
    def classifier_node(state: State) -> dict:
        """Look at the latest user message and decide which path to take"""
        last_message = state['messages'][-1].content
        prompt = (
            "Classify the following student query into exactly one category: "
            "'academic', 'fee', or 'general'.\n\n"
            "Use 'academic' for questions about attendance, exams, grading, credits"
            "promotion, course structure, summer training, or degree requirements.\n"
            "Use 'fee' for questions about tuition, payment, refund, late charges, "
            "scholarships, or any money-related topic. \n"
            "Use 'general' for greetings, casual talk, or anything not related to "
            "the college rules or fee.\n\n"
            f"Query: {last_message}\n\n"
            "Return only one word: academic, fee, or general."
        )

        response = llm.invoke(prompt)
        category = response.content.strip().lower()

        if 'academic' in category:
            category = 'academic'
        elif 'fee' in category:
            category = 'fee'
        else:
            category = 'general'

        return {'query_type': category}

    def academic_rag_nodes(state: State) -> dict:
        """Retrieves relevant chunks from academic handbook"""
        query = state["messages"][-1].content
        docs = academic_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    def fee_rag_node(state: State) -> dict:
        """Retrieves relevant chunks from fee structure PDF"""
        query = state["messages"][-1].content
        docs = fee_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    def general_node(state: State) -> dict:
        """Answers directly using the LLM's own knowledge, no retrieval needed."""
        return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

    def response_node(state: State) -> dict:
        """Generates the final answer, personalized using the student's programme"""
        query = state['messages'][-1].content
        programme = state.get('programme', 'unknown')
        context = state['retrieved_context']

        if context == 'NO_RETRIEVAL_NEEDED':
            prompt = (
                f'You are a friendly college assistant talking to a {programme} student'
                f'Answer this question using your own general knowledge:\n\n{query}'
            )
        else:
            prompt = (
                f"You are a college assistant helping a {programme} student. "
                f"Use the following context from the official college documents to answer"
                f"the question accurately. If the context mentions specific figures for "
                f"different programmes, highlight the one relevant to {programme}"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Give a clear, friendly, and precise answer."
            )
        response = llm.invoke(prompt)
        return {"messages": [("ai", response.content.strip())]}

    # ============================================================
    # Step-4 Router Function
    # ============================================================
    def route_query(state: State):
        if state["query_type"] == 'academic':
            return 'academic_rag'
        elif state['query_type'] == 'fee':
            return 'fee_rag'
        else:
            return 'general'

    # ============================================================
    # Step-5 Building the graph
    # ============================================================
    graph = StateGraph(State)
    graph.add_node('classifier', classifier_node)
    graph.add_node('academic_rag', academic_rag_nodes)
    graph.add_node('fee_rag', fee_rag_node)
    graph.add_node('general', general_node)
    graph.add_node('response', response_node)

    graph.add_edge(START, 'classifier')
    graph.add_conditional_edges('classifier', route_query)
    graph.add_edge('academic_rag', 'response')
    graph.add_edge('fee_rag', 'response')
    graph.add_edge('general', 'response')
    graph.add_edge('response', END)

    return graph.compile()


app = build_app()

# ============================================================
# Streamlit UI
# ============================================================

st.markdown("""
<style>
.stChatMessage { border-radius: 12px; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

if "programme" not in st.session_state:
    st.session_state.programme = "BCA"

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}

with st.sidebar:
    st.title("🎓 College Assistant")
    st.markdown("Ask me about **academics**, **fees**, or anything general!")

    st.divider()

    programme_choice = st.radio(
        "Select your programme:",
        options=["BCA", "BBA", "B.Com (H)"],
        index=["BCA", "BBA", "B.Com (H)"].index(st.session_state.programme),
    )
    st.session_state.programme = programme_choice

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Currently set as: **{st.session_state.programme} student**")

st.title("💬 Ask the College Assistant")
st.caption(f"Personalized for **{st.session_state.programme}** students")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Type your question here...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = app.invoke({
                    "programme": st.session_state.programme,
                    "messages": [("human", user_query)],
                })
                answer = result["messages"][-1].content
            except Exception as e:
                answer = f"⚠️ Something went wrong: {e}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})