from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

file_path = r"E:\Workspace\ai\RAG\内容加载与切片\RAG+LangGraph+Milvus.md"
# loader = UnstructuredMarkdownLoader(
#     file_path=file_path,
#     mode="single", # 如果使用"single"模式，文档将作为单个 Document 对象返回
#     strategy="fast",
# )

loader = UnstructuredMarkdownLoader(
    file_path=file_path,
    mode="elements",                         # 如果使用elements模式 unstructured 库会将文档拆分为诸如 Title 和 NarrativeText 等元素
    strategy="fast",
)

# 普通load方法
# docs = loader.load()

# 懒加载方法
# 保存为json文件 路径 + 保存文件名file_name
def write_json(data, file_name):
    with open(r'E:\Workspace\ai\RAG\内容加载与切片\mdoutput' + '\\' + file_name, 'w', encoding='utf-8') as f:
        import json
        json.dump(data, f, ensure_ascii=False, indent=4)           # indent=4 制表符个空格


docs = []
counter = 0
for doc in loader.lazy_load():
    docs.append(doc)
    json_file_name = str(counter) + ".json"
    counter += 1
    # write_json(doc.model_dump(), json_file_name)


assert isinstance(docs[0], Document)       # 断言第一个元素是一个 Document 实例。
print(f"文档总页数: {len(docs)}")
print(f"第5页元数据:")
print(docs[5].metadata)
print(f"第5页内容:")
print(docs[5].page_content[:150])        # 只打印前150个字符，避免内容过长
