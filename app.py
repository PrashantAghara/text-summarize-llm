import validators
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from urllib.parse import urlparse, parse_qs


def extract_video_id(url):
    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path[1:]
    if parsed.hostname and "youtube.com" in parsed.hostname:
        return parse_qs(parsed.query).get("v", [None])[0]
    return None


def load_youtube_docs_chunked(url):
    video_id = extract_video_id(url)
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)
    docs = []
    for item in transcript:
        docs.append(
            Document(
                page_content=item.text,
                metadata={
                    "source": url,
                    "start": item.start,
                    "duration": item.duration,
                },
            )
        )
    return docs


# Streamlit App
st.set_page_config(page_title="Summarize text from Websites")
st.title("Summarize text from your Websites")
st.subheader("Summarize URL")

with st.sidebar:
    api_key = st.text_input("GROQ API Key", value="", type="password")

url = st.text_input("URL", label_visibility="collapsed")

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key, streaming=True)

prompt_template = """
Provide a summary of the following content in 300 words :
{text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize"):
    if not api_key.strip() or not url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Waiting..."):
                # loading the data
                if "youtube.com" in url:
                    docs = load_youtube_docs_chunked(url)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        },
                    )
                    docs = loader.load()
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                summary = chain.run(docs)

                st.success(summary)
        except Exception as e:
            print(e)
            st.exception(f"Exception : {e}")
