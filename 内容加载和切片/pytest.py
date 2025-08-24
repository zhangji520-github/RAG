from langchain_community.document_loaders import PyPDFLoader
import sys
import os
from env_utils import OPENAI_API_KEY

file_path="/home/ji294/AI/RAG/内容加载和切片/一种基于RBF 神经网络的多无人艇包含控制策略.pdf"
loader = PyPDFLoader(file_path)

pages=[]

async def load_and_print():
    async for page in loader.alazy_load():
        pages.append(page)
    print(f"{pages[0].metadata}\n")
    print(f'doc的数量为: {len(pages)}\n')
    # print(pages[0].page_content)


if __name__ == "__main__":
    import asyncio
    asyncio.run(load_and_print())
