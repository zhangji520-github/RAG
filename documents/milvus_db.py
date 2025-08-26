from pymilvus import IndexType
from pymilvus.client.types import MetricType
from llm_utils import qwen_embeddings
from langchain_core.documents import Document




class MilvusVectorSave:
    """把新的document数据插入到数据库"""

    def __init__(self):
        """放一些索引的配置参数"""
        self.params = [
            {
                "field_name": "sparse",
                "index_name": "sparse_inverted_index",
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",
                "params": {
                    "inverted_index_algo": "DAATA_MAXSCORE", # （默认）：采用 MaxScore 算法优化的文档逐次（DAAT）查询处理。
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


    def create_collection(self, collection_name: str, dimension: int):
        """创建一个collection milvus + langchain"""
        # 检查集合是否已存在，如果存在则删除
        if client.has_collection("my_collection"):
            client.drop_collection("my_collection")
            print("已删除现有集合 'my_collection'")

    def add_documents(self, collection_name: str, documents: list[Document]):
        """添加新的document数据到数据库"""
        pass