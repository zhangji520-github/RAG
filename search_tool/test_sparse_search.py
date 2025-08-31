import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_milvus import BM25BuiltInFunction, Milvus
from pymilvus import MilvusClient, DataType
from env_utils import MILVUS_URI
from pymilvus import Function, FunctionType
from llm_utils import openai_embedding
from langchain_core.documents import Document
from documents.markdown_parser import MarkdownParser
"""
测试 Milvus 全文检索
"""

def sparse_search():
    # 1. 创建一个 Milvus 连接
    client = MilvusClient(uri=MILVUS_URI)
    print("Milvus 数据库连接已建立")
    # 2. 定义schema
    schema = client.create_schema()

    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("category", DataType.VARCHAR, max_length=1000)
    # schema.add_field("text", DataType.VARCHAR, max_length=6000, enable_analyzer=True, analyzer_params={"type": "chinese"}) # 注意如果语料是中文，需要开启分词器为中文
    schema.add_field("text", DataType.VARCHAR, max_length=6000, enable_analyzer=True, analyzer_params={"tokenizer": "jieba"}) # 注意如果语料是中文，需要开启分词器为中文
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

    # 3 构建用于全文搜索的 bm25 Function
    # bm25用于将文本字段转化为稀疏向量字段 用于全文检索
    bm25_function = Function(
        name = "text_bm25_emb",
        input_field_names=["text"], # # 需要进行文本到稀疏向量转换的 VARCHAR 字段名称。对于 FunctionType.BM25 ，此参数仅接受一个字段名称。
        output_field_names=["sparse"], # # 存储内部生成稀疏向量的字段名称。对于 FunctionType.BM25 ，此参数仅接受一个字段名。
        function_type=FunctionType.BM25 # # 要使用的函数类型。将该值设置为 FunctionType.BM25
    )
    schema.add_function(bm25_function)
    # 配置索引
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="sparse",

        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE", # （默认）：采用 MaxScore 算法优化的文档逐次（DAAT）查询处理。
            "bm25_k1": 1.2,       # 数值越高，专业术语词频在文档排名中的重要性越大。取值范围：[1.2, 2.0]。
            "bm25_b": 0.75 
        }
    )

    # 4. 创建集合
    # 检查集合是否已存在，如果存在则删除
    if client.has_collection("my_collection"):
        client.drop_collection("my_collection")
        print("已删除现有集合 'my_collection'")

    # 创建集合collections
    client.create_collection(
        collection_name="my_collection",
        schema=schema,
        index_params=index_params,
    )

# 使用 Milvus 自带的schema处理更加灵活  这里我们用langchain_milvus的Milvus来作为存储向量数据库
def insert_data():
    """插入测试数据"""
    vector_store = Milvus(
        embedding_function=None,
        collection_name="my_collection",
        connection_args={
            "uri": MILVUS_URI,
        },
        builtin_function=BM25BuiltInFunction( 
            input_field_names="text",      # 输入：原始文本字段 	VARCHAR
            output_field_names="sparse",   # 输出：稀疏向量字段，对应 vector_field[0] SPARSE_FLOAT_VECTOR
        ),      
        vector_field=["sparse"],
        consistency_level="Strong",
    )

    file_path = r"E:\Workspace\ai\RAG\datas\md\tech_report_z7tx05vt.md"
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)         # 读取多个文档

    vector_store.add_documents(docs)   # 插入数据

    return vector_store

if __name__ == "__main__":
    sparse_search()
    insert_data()

    vector_store = insert_data()
    print("\n使用过滤条件进行搜索:")
    result = vector_store.similarity_search_with_score(
        query = "湿法刻蚀的优势", 
        k=2,
        expr='category == "TitleWithContent"',
        consistency_level="Eventually",  
        )
    for doc, score in result:
        print(f"内容: {doc.page_content}")
        print(f"相似度得分: {score}")
        print("-" * 30)