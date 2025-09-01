from pymilvus import MilvusClient
from env_utils import COLLECTION_NAME, MILVUS_URI

client = MilvusClient(uri=MILVUS_URI)
if client.has_collection(COLLECTION_NAME):
    schema = client.describe_collection(COLLECTION_NAME)
    print('集合 schema 信息:')
    for field in schema['fields']:
        print(f'字段名: {field["name"]}, 类型: {field["type"]}')
else:
    print(f'集合 {COLLECTION_NAME} 不存在')
