import streamlit as st

from langchain_groq import ChatGroq
from langchain_classic.chains.llm_math.base import LLMMathChain
from langchain_classic.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType, Tool
from langchain_classic.callbacks import StreamlitCallbackHandler

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Math Problem Solver", page_icon="🧠")
st.title("🧠 Text to Math Problem Solver")

# ---------------------------------------------------
# API KEY
# ---------------------------------------------------
groq_api_key = st.sidebar.text_input(
    "Enter Groq API Key",
    type="password"
)

if not groq_api_key:
    st.info("Please enter your Groq API key in sidebar.")
    st.stop()

# ---------------------------------------------------
# LLM
# ---------------------------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
    temperature=0
)

# ---------------------------------------------------
# TOOLS
# ---------------------------------------------------

# Wikipedia Tool
wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=250
    )
)

# Search Tool
search = DuckDuckGoSearchRun(name="Search")

# Calculator Tool
math_chain = LLMMathChain.from_llm(
    llm=llm,
    verbose=False
)

calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Useful for solving mathematical calculations."
)

# Reasoning Tool
prompt_template = """
You are an expert reasoning assistant.

Solve the user question clearly and step-by-step.

Question: {question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=prompt_template
)

reasoning_chain = LLMChain(
    llm=llm,
    prompt=prompt
)

reasoning_tool = Tool(
    name="ReasoningTool",
    func=reasoning_chain.run,
    description="Useful for logic, reasoning, and word problems."
)

# ---------------------------------------------------
# AGENT
# ---------------------------------------------------
agent = initialize_agent(
    tools=[calculator, reasoning_tool, wiki, search],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi 👋 Ask me any math problem."
        }
    ]

# Show previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------------------------------------------
# INPUT
# ---------------------------------------------------
question = st.text_area(
    "Enter your question:",
    "What is area of circle with radius 4?"
)

# ---------------------------------------------------
# BUTTON
# ---------------------------------------------------
if st.button("Find Answer"):

    if question.strip():

        # Show user message
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )
        st.chat_message("user").write(question)

        with st.spinner("Generating answer..."):

            try:
                st_cb = StreamlitCallbackHandler(
                    st.container(),
                    expand_new_thoughts=False
                )

                response = agent.invoke(
                    {"input": question},
                    config={"callbacks": [st_cb]}
                )

                answer = response["output"]

            except Exception as e:
                answer = f"Error: {str(e)}"

        # Show assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        st.chat_message("assistant").write(answer)

    else:
        st.warning("Please enter a question.")
