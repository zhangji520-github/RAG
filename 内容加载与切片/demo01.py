from langchain_core.documents import Document


def load_doc_from_json(json_file):
    """从json文件加载为Document对象"""
    with open(json_file, 'r', encoding='utf-8') as f:
        import json
        data = json.load(f)         # 加载JSON数据
        return Document(page_content=data["page_content"], metadata=data["metadata"])    # 创建Document对象
    

if __name__ == "__main__":
    file_path = r'E:\Workspace\ai\RAG\内容加载与切片\output\1_0.json'
    doc = load_doc_from_json(file_path)
    print(f"元数据: {doc.metadata}/n")
    print(f"内容: {doc.page_content}")