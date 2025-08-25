import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from utils.log_utils import log
from typing import List
from llm_utils import openai_embedding
from langchain_experimental.text_splitter import SemanticChunker

class MarkdownParser:
    """
    Markdown解析器 处理解析与切片
    """
    
    def __init__(self):
        self.text_splitter = SemanticChunker(
            openai_embedding, breakpoint_threshold_type="percentile"
        )
        
    def text_chunker(self, datas: List[Document]) -> List[Document]:
        new_docs = []
        for d in datas:
            if len(d.page_content) > 5000:  # 内容超出了阈值，则按照语义再切割
                new_docs.extend(self.text_splitter.split_documents([d]))
                continue
            new_docs.append(d)
        return new_docs

    def parse_markdown_to_documents(self, md_file_path: str, encoding: str = "utf-8") -> list[Document]:
        """
        解析Markdown文件为Document对象列表
        :param md_file_path: Markdown文件路径
        :param encoding: 文件编码，默认utf-8
        :return: Document对象列表
        """
        documents = self.parse_markdown(md_file_path, encoding)
        log.info(f"解析Markdown文件 {md_file_path} 成功，解析得到 {len(documents)} 个Document对象")

        merged_documents = self.merge_title_content(documents)
        log.info(f"合并标题和内容后，得到 {len(merged_documents)} 个Document对象")

        # 切分文本
        chunk_documents = self.text_chunker(merged_documents)
        log.info(f"文本切分后，得到 {len(chunk_documents)} 个Document对象")

        return  chunk_documents  # 添加返回语句

    def parse_markdown(self, md_file_path: str, encoding: str = "utf-8") -> list[Document]:
        """
        解析Markdown文件为Document对象列表
        :param md_file_path: Markdown文件路径
        :param encoding: 文件编码，默认utf-8
        :return: Document对象列表
        """
        loader = UnstructuredMarkdownLoader(
            file_path=md_file_path,
            mode="elements",  # 使用elements模式，文档将拆分为诸如 Title 和 NarrativeText 等元素
            strategy="fast",
            encoding=encoding
        )
        docs = []
        for doc in loader.lazy_load():
            docs.append(doc)

        return docs
    def merge_title_content(self, datas: List[Document]) -> List[Document]:
        """
        这个函数的主要目的是重组使用 unstructured 库解析的 Markdown 文档，
        将标题和内容按照层级结构进行合并。通过维护一个父元素字典，
        它能够构建出包含完整上下文信息的文档结构，
        使得每个内容块都包含了其相关的标题信息，从而在后续的检索或处理中能够获得更完整的语义信息。
        :param datas: Document对象列表
        :return: 合并后的Document对象列表
        """
        merged_data = []
        parent_dict = {}  # parent_dict 是一个字典（dictionary），用于存储所有标题类型的文档（category 为 'Title' 的文档）。
        for document in datas:
            metadata = document.metadata
            if 'languages' in metadata:          # 获取每个文档的元数据，并移除其中的 'languages' 字段（可能是因为这个信息不重要或会干扰处理
                metadata.pop('languages')

            # 从document中获取元数据信息 parent_id, category, element_id
            parent_id = metadata.get('parent_id', None)
            category = metadata.get('category', None)
            element_id = metadata.get('element_id', None)
            # 条件1：独立的内容文档（没有父级的NarrativeText）
            if category == 'NarrativeText' and parent_id is None:  # 是否为：内容document 直接添加到merged_data中
                merged_data.append(document)
            # 条件2：标题文档
            if category == 'Title':    # 若是标题，用一个'Title'字段保存在document['metadata']中
                document.metadata['title'] = document.page_content # Document里面就metadata跟page_content，我们直接在metadata中添加title字段 内容
                # 检查parent_id是否存在于parent_dict中，避免KeyError
                if parent_id and parent_id in parent_dict:
                    document.page_content = parent_dict[parent_id].page_content + ' -> ' + document.page_content
                parent_dict[element_id] = document  # parent_id key为element_id，value为document 包含新的字段 title 
            # 条件3：有父级的内容文档  遇到非标题类型（如 NarrativeText、ListItem 等）且有父元素ID的文档时 将该文档的内容追加到其父标题文档的内容中
            # 将父文档的类别标记为 'content'，表示它现在包含了内容信息
            if category != 'Title' and parent_id:
                # 检查parent_id是否存在于parent_dict中，避免KeyError
                if parent_id in parent_dict:
                    parent_dict[parent_id].page_content = parent_dict[parent_id].page_content + ' ' + document.page_content
                    parent_dict[parent_id].metadata['category'] = 'content'
                else:
                    # 如果找不到父文档，将当前文档作为独立文档添加到结果中
                    log.warning(f"找不到parent_id为 {parent_id} 的父文档，将当前文档作为独立文档处理")
                    merged_data.append(document)
    
        # 将parent_dict中包含内容的文档添加到merged_data
        for doc in parent_dict.values():
            if doc.metadata.get('category') == 'content':
                merged_data.append(doc)
        
        return merged_data

    def get_logger(self):
        return self.logger

if __name__ == "__main__":
    file_path = r"E:\Workspace\ai\RAG\datas\md\tech_report_0tfhhamx.md"
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)

    counter = 1
    for item in docs:
        print(f'第 {counter} 页')
        print(f'元数据: {item.metadata}')
        print("-----" * 10)
        print(f'标题: {item.metadata.get("title", "无标题")}')    # 正常标题
        print("-----" * 10)
        # 打印当前counter页的内容 这里我们把主标题和子标题都放在内容中
        print(f'内容: {item.page_content}')
        print("-----" * 10)
        counter += 1