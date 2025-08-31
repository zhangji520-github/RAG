import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from documents.milvus_db import MilvusVectorSave


if __name__ == "__main__":
    """测试 搜索功能"""
    # 1. 创建一个 Milvus 连接
    mv = MilvusVectorSave()
    mv.create_connection(is_first=False)   # 前面已经将数据加到集合中了
    print("Milvus 数据库连接已建立，集合已创建")
    # 2. 测试搜索功能
    # 也可以转换为检索器进行检索
    # retriever = mv.vector_stored_saved.as_retriever(
    #     search_type="similarity",
    #     search_kwargs={"k": 2}  # 返回最相似的2个文档块
    # )
    # print("\n使用检索器进行搜索:")
    # for i,doc in enumerate(retriever.invoke("先进纳米级清洗技术？"), 1):
    #     print(f"结果 {i}: {doc.page_content}")
    #     print("-" * 30)


    # 按照标量字段category进行过滤搜索 如果您需要完全控制，可以绕过 LangChain 的封装，直接使用 pymilvus 客户端调用 search 方法，这样就可以直接使用 output_fields anns_field参数。
    print("\n使用过滤条件进行搜索:")
    result = mv.vector_stored_saved.similarity_search_with_score(
        query = "先进纳米级清洗技术？", 
        k=2,
        expr='category == "TitleWithContent"',
        consistency_level="Eventually",  
        )
    for doc, score in result:
        print(f"内容: {doc.page_content}")
        print(f"相似度得分: {score}")
        print("-" * 30)