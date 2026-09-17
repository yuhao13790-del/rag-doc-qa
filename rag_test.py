import os
import chromadb
from chromadb.utils import embedding_functions

# 1. 指定路径
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DATA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

# 2. 初始化 Chroma 数据库
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# 3. 使用默认的向量化模型（第一次运行会自动下载模型，请耐心等待）
print("正在初始化模型，第一次运行可能需要下载文件...")
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_or_create_collection(name="my_knowledge", embedding_function=embedding_fn)

# 4. 读取 data 文件夹里的 txt 文件并存入
documents = []
metadatas = []
ids = []

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".txt"):
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        # 按行切块
        chunks = [chunk.strip() for chunk in content.split("\n") if chunk.strip()]
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": filename, "chunk_index": i})
            ids.append(f"{filename}_{i}")

print(f"✅ 成功读取 {len(documents)} 个文本块，正在存入向量数据库...")

# 先清空旧数据（避免重复运行时报错）
existing = collection.get()
if existing['ids']:
    collection.delete(ids=existing['ids'])

# 存入数据库
collection.add(documents=documents, metadatas=metadatas, ids=ids)
print("✅ 存入完毕！")

# 5. 测试检索！
query = "请假一天扣多少钱？"
print(f"\n🔍 模拟用户提问: {query}")

results = collection.query(
    query_texts=[query],
    n_results=2
)

print("\n🎯 检索到的相关资料:")
for i, doc in enumerate(results['documents'][0]):
    print(f"【资料 {i+1}】: {doc}")