import sys
import os

from numpy import doc
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiprocessing import Process, Queue
import time, os
from utils.log_utils import log
from markdown_parser import MarkdownParser
# 采用多进程 分布式 的方式把海量的数据写入 Milvus 数据库 建立一个共享的队列(内部维护着数据的共享)，多个进程可以向队列里存/取数据

def file_parser_process(dir_path: str, output_queue: Queue, batch_size: int = 10):
    """进程1: 解析目录下所有的md文件并分批放入到队列中"""
    log.info(f"文件解析进程启动，解析目录: {dir_path}")

    # 获取目录下所有的md文件 使用 os.path.join() 将目录路径和文件名组合成完整的文件路径
    md_files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)   # 遍历 dir_path 目录下的所有文件和文件夹的名称列表
        if f.endswith(".md")
    ]

    if not md_files:
        log.warning(f"目录 {dir_path} 下没有找到任何 Markdown 文件。")
        output_queue.put(None)    # 发送结束信号
        return
    
    parser = MarkdownParser()  # 将 doc 转化为 Document 对象

    doc_batch = []    # 缓冲区 临时存储从 Markdown 文件解析出来的 Document 对象。
    # 典型的“流式批处理”模式
    for file_path in md_files:
        try:
            documents = parser.parse_markdown_to_documents(file_path)
            if documents:
                doc_batch.extend(documents)  # 将解析得到的 Document 对象添加到缓冲区 数组加到数组用 extend

            # 如果缓冲区达到批量大小，则将其放入队列并清空缓冲区
            if len(doc_batch) >= batch_size:
                output_queue.put(doc_batch.copy())  # 放入队列时使用 copy 避免引用问题 把当前批次发走
                doc_batch.clear()   # 清空缓冲区的所有批次数据 清空，准备下一批
                log.info(f"已将 {batch_size} 个 Document 对象放入队列")

        except Exception as e:
            log.exception(f"解析文件 {file_path} 时出错，上下文信息：文件大小={os.path.getsize(file_path)}字节, 当前批次大小={len(doc_batch)}", exc_info=e)
    
    # 继续发送剩余的documents
    if doc_batch:
        output_queue.put(doc_batch)
        log.info(f'解析完成，共处理 {len(md_files)} 个文件')
        