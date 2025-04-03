import hnswlib
import numpy as np
import time

# 打印文件信息，检查数据是否符合预期
def inspect_fvecs_file(path):
    with open(path, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0, 2)  # 移动到文件末尾
        file_size = f.tell()
        num_vectors = file_size // ((dim + 1) * 4)
        print(f"File: {path}, Dimension: {dim}, Total Vectors: {num_vectors}")
        print(f"Expected Size: {num_vectors * (dim + 1) * 4} bytes, Actual Size: {file_size} bytes")


# 读取 .fvecs 文件
def load_fvecs(path):
    with open(path, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]  # 读取第一个维度信息
        f.seek(0, 2)  # 移动到文件末尾
        file_size = f.tell()
        num_vectors = file_size // ((dim + 1) * 4)  # 总向量数
        f.seek(0, 0)  # 返回文件开头

        data = np.zeros((num_vectors, dim), dtype=np.float32)  # 初始化空数组
        for i in range(num_vectors):
            f.read(4)  # 跳过维度信息
            data[i] = np.fromfile(f, dtype=np.float32, count=dim)  # 读取实际向量数据
    return data

# 读取 .ivecs 文件
def load_ivecs(path):
    with open(path, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]  # 读取第一个维度信息
        f.seek(0, 2)  # 移动到文件末尾
        file_size = f.tell()
        num_vectors = file_size // ((dim + 1) * 4)  # 计算总向量数
        f.seek(0, 0)  # 返回文件开头

        data = np.zeros((num_vectors, dim), dtype=np.int32)  # 初始化数组
        for i in range(num_vectors):
            f.read(4)  # 跳过维度信息
            data[i] = np.fromfile(f, dtype=np.int32, count=dim)  # 读取实际向量数据
    return data

# 计算召回率
def calculate_recall(index, queries, ground_truth, k, ef_search):
    index.set_ef(ef_search)  # 设置搜索时的 ef 参数
    labels, _ = index.knn_query(queries, k=k)
    correct = 0
    total = queries.shape[0] * k

    for i in range(queries.shape[0]):
        gt_set = set(ground_truth[i, :k])
        correct += sum(1 for idx in labels[i] if idx in gt_set)
    
    return correct / total

# 主函数
def main():
    # 参数
    M = 16
    ef_construction = 200
    ef_search = 200
    k = 50
    num_threads = 8  # 设置多线程数目

    # 文件路径
    path_sift_base = "./sift/sift_base.fvecs"
    path_sift_query = "./sift/sift_query.fvecs"
    path_sift_groundtruth = "./sift/sift_groundtruth.ivecs"

    # 开始计时
    start_time = time.time()

    inspect_fvecs_file("./sift/sift_base.fvecs")

    # 加载基向量、查询向量和真实集
    base_data = load_fvecs(path_sift_base)
    query_data = load_fvecs(path_sift_query)
    ground_truth = load_ivecs(path_sift_groundtruth)
    
    print(f"Loaded SIFT1M dataset. Base size: {base_data.shape}, Query size: {query_data.shape}")

    # 构建 HNSW 索引
    dim = base_data.shape[1]
    num_elements = base_data.shape[0]
    index = hnswlib.Index(space='l2', dim=dim)  # 使用 L2 距离
    index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
    
    # 使用多线程添加数据
    index.add_items(base_data, np.arange(num_elements), num_threads=num_threads)
    print("Index built successfully with multi-threading!")

    # 计算召回率
    print("Calculating recall...")
    recall = calculate_recall(index, query_data, ground_truth, k, ef_search)
    print(f"Recall @ k={k}: {recall}")

    # 计算处理时间
    elapsed_time = time.time() - start_time
    print(f"处理时间: {elapsed_time * 1000:.2f} 毫秒")

if __name__ == "__main__":
    main()
