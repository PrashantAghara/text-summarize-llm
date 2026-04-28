import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.chains.llm_math.base import LLMMathChain
from langchain_classic.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType, Tool
from langchain_classic.callbacks import StreamlitCallbackHandler

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
)
search = DuckDuckGoSearchRun(name="Search")

st.set_page_config(page_title="Text to Math problem solver")
st.title("Text to Math problem solver")

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please provide your GROQ API Key")
    st.stop()

llm = ChatGroq(model="qwen/qwen3-32b", groq_api_key=groq_api_key)

# Tools
math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Tool to answer math problems. Only input mathematical expression needs to be provided",
)

prompt = """
You are a agent for solving user's mathematical questions. 
Logically arrive at a solution and display it with point wise for the question below
Question : {question}
Answer:
"""

template = PromptTemplate(input_variables=["question"], template=prompt)

# Combine all the tools into chain
chain = LLMChain(llm=llm, prompt=template)
reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,
    description="Tool for answering logic based and reasoning questions.",
)


# Agents
agent = initialize_agent(
    tools=[wiki, reasoning_tool, calculator, search],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_error=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I can answer your any Math Problem"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# Lets start the interaction
question = st.text_area(
    "Enter your question: ", "What is area of circle with radius 4?"
)

if st.button("Find Answer"):
    if question:
        with st.spinner("Generate response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write("### Response:")
            st.success(response)
    else:
        st.warning("Please enter your question")
