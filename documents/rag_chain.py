import sys
import os
# 添加上级目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from llm_utils import qwen_embeddings, openai_embedding, llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from milvus_db import MilvusVectorSave
from langchain.prompts import PromptTemplate
from markdown_parser import MarkdownParser


# Define the prompt template for generating AI responses
PROMPT_TEMPLATE = """
Human: You are an AI assistant, and you provide answers to questions by using fact based and statistical information when possible.
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

# Create a PromptTemplate instance with the defined template and input variables
prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

class RagChain:
    """自定义ragchain"""
    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def run_chain(self, retrieval, question):
        # 首先获取检索结果
        retrieved_docs = retrieval.invoke(question)
        
        # 打印检索结果
        print("🔍 检索结果:")
        print("=" * 40)
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"文档片段 {i}:")
            print(f"内容: {doc.page_content}")
            print("-" * 30)
        
        # 构建RAG链
        rag_chain = (
            {"context": retrieval | RagChain.format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # 生成答案
        print("\n🤖 生成答案:")
        # res = rag_chain.invoke(question)
        # print(res)
        for chunk in rag_chain.stream(question):
            print(chunk, end="", flush=True)

if __name__ == "__main__":
    # 1. 加载文档数据
    file_path = r"E:\Workspace\ai\RAG\datas\md\tech_report_z7tx05vt.md"
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)
    print(f"加载了 {len(docs)} 个文档块")

    # 2. 创建 Milvus 向量数据库并添加文档 
    mv = MilvusVectorSave()
    mv.create_connection(is_first=True)
    mv.add_documents(docs)
    print("文档已添加到向量数据库")

    # 3. 创建 RAG 链并测试
    rag = RagChain()
    
    # 4. 设置检索器 - 使用向量相似度搜索
    retriever = mv.vector_stored_saved.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1}  # 返回最相似的1个文档块
    )
    
    # 5. 测试问题
    test_questions = [
        "干法刻蚀的优势？",
    ]
    
    print("\n开始测试 RAG 系统:")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 30)
        
        try:
            rag.run_chain(retriever, question)
        except Exception as e:
            print(f"错误: {e}")
        
        print("-" * 50)
