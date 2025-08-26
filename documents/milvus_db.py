import sys
import os
# 添加上级目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pymilvus import IndexType, MilvusClient
from pymilvus.client.types import MetricType
from llm_utils import qwen_embeddings, openai_embedding, llm
from langchain_core.documents import Document
from env_utils import MILVUS_URI, COLLECTION_NAME
from langchain_milvus import Milvus, BM25BuiltInFunction
from typing import List, Optional
from markdown_parser import MarkdownParser


class MilvusVectorSave:
    """把新的document数据插入到数据库"""

    def __init__(self):
        # 类型注解：明确声明属性类型，提供IDE智能提示和类型检查
        self.vector_stored_saved: Optional[Milvus] = None
        """放一些索引的配置参数 __init__ 应该只负责初始化状态，而不是执行“创建集合”这种业务动作  注意配置好了字段参数vector_field之后一定要配置对应的索引参数index_params"""
        self.params = [
            {
                "field_name": "sparse",
                "index_name": "sparse_inverted_index",
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",
                "params": {
                    "inverted_index_algo": "DAAT_MAXSCORE", # （默认）：采用 MaxScore 算法优化的文档逐次（DAAT）查询处理。
                    "bm25_k1": 1.2,       # 数值越高，专业术语词频在文档排名中的重要性越大。取值范围：[1.2, 2.0]。
                    "bm25_b": 0.75 
                }
            },
            {
                "field_name": "dense",
                "index_name": "dense_vector_index",        # 向量索引 HNSW 矢量索引
                "index_type": IndexType.HNSW,              # 矢量索引类型  也可以直接用"AUTOINDEX" 这样就省的配置参数了
                "metric_type": MetricType.IP,         # 用于计算向量间距离的方法。支持的取值包括 COSINE 、 L2 和 IP 。详情请参阅度量类型。
                "params":{
                    "M": 16,            # 数值越大，精度越高但内存消耗越大
                    "efConstruction": 100 # 索引构建过程中考虑连接的候选邻居数量 数值越大，索引质量越高但构建时间越长
                }
            }
        ]

    def create_collection(self,collection_name: str = COLLECTION_NAME, uri: str = MILVUS_URI, is_first: bool = False):
        """创建一个collection milvus + langchain"""
        # 检查集合是否已存在，如果存在先释放collection，然后再删除索引和集合  

        if is_first:  
            client = MilvusClient(uri=MILVUS_URI)
            if COLLECTION_NAME in client.list_collections():
                client.release_collection(collection_name=COLLECTION_NAME)
                client.drop_index(collection_name=COLLECTION_NAME, index_name="sparse_inverted_index")
                client.drop_index(collection_name=COLLECTION_NAME, index_name="dense_vector_index")
                client.drop_collection(collection_name=COLLECTION_NAME)

        # 利用langchain提供的milvus 工具创建存储向量的collection
        # BM25BuiltInFunction() 是专门为 LangChain 的 Milvus.from_documents() 方法设计的
        self.vector_stored_saved = Milvus(
            embedding_function=qwen_embeddings,
            collection_name=collection_name,
            connection_args={
                "uri": uri,
            },
            # 自动将文本字段（TEXT_FIELD）通过 内置的 BM25 函数 转换为 稀疏向量（SPARSE_FLOAT_VECTOR）
            builtin_function=BM25BuiltInFunction( 
                input_field_names="text",      # 输入：原始文本字段 	VARCHAR
                output_field_names="sparse",   # 输出：稀疏向量字段，对应 vector_field[0] SPARSE_FLOAT_VECTOR
            ),           
            vector_field=["sparse","dense"],  # 指定要存储的向量字段 默认稠密字段用的是"vector",稀疏字段用的是"sparse" 。这里我们自定义  注意默认还有 pk字段 和 text 字段
            index_params=self.params,          # 索引参数   vector_field和index_params的配置参数要一一对应，不能混淆
            consistency_level="Strong",          # 一致性级别
            auto_id=True,                       # 是否主键自动生成ID   
        )
    def add_documents(self, documents: list[Document]):
        """
        添加新的 document 数据到数据库
        :param documents: LangChain Document 列表
        """
        # langchain会自动 把 doc.page_content 的值，赋给 Milvus 中叫 `text` 的字段
        # 1.调用内置的 BM25BuiltInFunction 然后 Milvus 用 analyzer 处理这个 `text` 字段，生成 sparse 向量
        # 2.调用 embedding_function 将 page_content 转换成 dense 向量 存到你定义的 "dense" 字段
        self.vector_stored_saved.add_documents(documents)

if __name__ == "__main__":
    file_path = r"E:\Workspace\ai\RAG\datas\md\tech_report_0ls2kg7u.md"
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)         # 读取多个文档

    mv = MilvusVectorSave()
    # 首先如果是第一次创建集合，就传 True
    mv.create_collection(is_first=True)

    mv.add_documents(docs)  # 添加新的 document 数据

    # 从向量存储中获取client
    client = mv.vector_stored_saved.client
    # 从client获取表结构并打印看看
    desc_collection = client.describe_collection(
        collection_name=COLLECTION_NAME
    )
    print(f"集合结构: {desc_collection}")
    print("-----" * 10)
    # 从client得到当前collection的所有索引index
    collection_Index = client.list_indexes(
        collection_name=COLLECTION_NAME
    )
    print(f"当前集合的索引: {collection_Index}")
    print("-----" * 10)
    # 打印索引列表的描述
    if collection_Index:
        for index_name in collection_Index:
            index_info = client.describe_index(
                collection_name=COLLECTION_NAME,
                index_name=index_name
            )
            print(f"索引 {index_name} 的描述: {index_info}")
    print("-----" * 10)

    # # 基于标量字段（如ID、字符串、数字）进行精确查询
    # filter = "category == 'Title'"
    # results = client.query(
    #     collection_name=COLLECTION_NAME,
    #     filter=filter,
    #     output_fields=["text", "sparse", "category", "filename"],
    #     limit=2  # 限制返回的结果数量
    # )
    # print(f"基于标量字段的查询结果: {results}")
    # print("-----" * 10)
    # # 基于向量字段进行向量查询
    # query = "什么可以有效提高深宽比结构的刻蚀精度和一致性"
    # query_vector = qwen_embeddings.embed_query(query)
    # search_params = {
    #     "params": {"nprobe": 10}
    # }
    # vector_results = client.search(
    #     collection_name=COLLECTION_NAME,
    #     data=[query_vector],  # 查询向量
    #     search_params=search_params,
    #     anns_field="dense",  # 向量字段名
    #     limit=2,  # 返回top2
    #     output_fields=["text", "category", "filename"]
    # )
    # print(f"基于向量字段的查询结果: {vector_results}")
    # print("-----" * 10)