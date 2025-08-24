from langchain_unstructured import UnstructuredLoader


file_path=r"E:\Workspace\ai\RAG\内容加载与切片\一种基于RBF 神经网络的多无人艇包含控制策略.pdf"
loader = UnstructuredLoader(
    file_path=file_path,
    strategy="hi_res",   # 解析策略，包括 "hi_res", "fast", "auto"
    partition_via_api=False,         # 使用本地的unstructured包，避免API付费问题
    # coordinates=True,        # 本地模式下移除此参数避免冲突
    # api_key="EbbLK1ZGPAoqz5OzJ7SOvmTXHp5UGc"  # 本地模式不需要API key
)

# 保存为json文件 路径 + 保存文件名file_name
def write_json(data, file_name):
    with open(r'E:\Workspace\ai\RAG\内容加载与切片\output' + '\\' + file_name, 'w', encoding='utf-8') as f:
        import json
        json.dump(data, f, ensure_ascii=False, indent=4)           # indent=4 制表符个空格


docs = []
counter = 0
for doc in loader.lazy_load():
    docs.append(doc)
    json_file_name = str(doc.metadata.get("page_number")) + "_" + str(counter) + ".json"
    counter += 1
    write_json(doc.model_dump(), json_file_name)

print(f"文档总页数: {len(docs)}")
print(f"第一页元数据:")
print(docs[0].metadata)
print(f"第一页内容:")
print(docs[0].page_content)