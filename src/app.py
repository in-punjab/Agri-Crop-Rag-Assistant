import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from retriever import load_vector_store
from langchain_core.messages import HumanMessage


# Load environment variables
load_dotenv()

# Page setup
st.set_page_config(page_title="Agri Crop Management Q&A", layout="centered")

st.title("🌱 Agri Crop Management Q&A Assistant")
st.markdown("### AI-powered crop management assistant for farmers")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



@st.cache_resource
def get_vectorstore():
    return load_vector_store()

@st.cache_resource
def load_llm():
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )



vectorstore = get_vectorstore()
llm = load_llm()


# PROMPT TEMPLATE

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an agricultural expert helping farmers.

IMPORTANT:
Always respond in clear professional English only.
Do NOT use Hindi, Punjabi, or any other language.

Context:
{context}

Question:
{question}

Provide concise, practical crop management advice in English:
"""
)



# DISPLAY CHAT HISTORY

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

user_question = st.chat_input("Ask about crop management...")

if user_question:

    # Show user message
    st.session_state.chat_history.append(("user", user_question))
    with st.chat_message("user"):
        st.write(user_question)

    with st.spinner("Generating agricultural advice..."):

        retriever = vectorstore.as_retriever(search_kwargs={"k":3})
        docs = retriever.invoke(user_question)


        context = "\n\n".join([doc.page_content for doc in docs])

        final_prompt = prompt.format(
            context=context,
            question=user_question
        )

        response = llm.invoke([
            HumanMessage(content=final_prompt)
        ])


        answer = response.content

    # Show assistant message

    with st.chat_message("assistant"):
        st.markdown("### 🌾 Recommended Farming Guidance")
        st.success(answer)


        # SHOW SOURCES 

        with st.expander("View Knowledge Sources"):
            for i, doc in enumerate(docs):
                st.write(f"Source {i+1}:")
                st.write(doc.page_content[:300])

    st.session_state.chat_history.append(("assistant", answer))
