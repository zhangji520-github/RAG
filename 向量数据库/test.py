import sys
import os

# 添加上级目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from llm_utils import openai_embedding,llm

loader = WebBaseLoader(
    web_paths=(
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

docs = text_splitter.split_documents(documents)

docs = docs[:10]  # 限制文档数量，避免批量过大报错
vectorstore = Milvus.from_documents(
    documents=docs,
    embedding=openai_embedding,
    connection_args={
        "uri": "http://localhost:19530",  # http://localhost:19530
    },
    collection_name="langchain_example",
    drop_old=True,  # Drop the old Milvus collection if it exists
)

# Define the prompt template for generating AI responses
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# 您是一个人工智能助手，在可能的情况下会基于事实和统计数据来回答问题。
# 请利用以下信息为<question>标签内的问题提供简洁回答。
# 若不知道答案，请直接说明不清楚，不要试图编造答案。
PROMPT_TEMPLATE = """
Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or numbers when possible.

Assistant:"""
prompt = PromptTemplate(
    template = PROMPT_TEMPLATE,
    input_variables = ["context", "question"]
)

# 构建检索器retrieval，便于集成到LCEL
retrieval = vectorstore.as_retriever()

# 定于检索器输出处理函数
def format_docs(docs):
    """Format the retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

query = "What is self-reflection of an AI Agent?"

rag_chain = (
    {"context": retrieval |  format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# rag_chain.get_graph().print_ascii()

# Invoke the RAG chain with a specific question and retrieve the response
# res = rag_chain.invoke(query)
# print(res)
# stream输出
for chunk in rag_chain.stream(query):
    print(chunk, end="", flush=True) 