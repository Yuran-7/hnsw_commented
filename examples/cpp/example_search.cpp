#include "../../hnswlib/hnswlib.h"


int main() {

    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();

    int dim = 8;               // Dimension of the elements
    int max_elements = 100000;  // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // Initing index
    hnswlib::L2Space space(dim);    // space_l2.h
    // 调用了第三个构造函数
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;  
    rng.seed(47);   // std::mt19937 rng(47);
    std::uniform_real_distribution<> distrib_real;  // 默认在[0.0, 1.0)之间
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }
#pragma omp parallel for
    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i);  // hnswalg.h第958行
    }

    // Query the elements for themselves and measure recall
    // 看看能不能找到自己是最近的
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {    // labeltype === unsigned long long
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";

    // Serialize index，将 HNSW 索引的内部数据结构序列化并保存到文件
    // std::string hnsw_path = "hnsw.bin";
    // alg_hnsw->saveIndex(hnsw_path);
    delete alg_hnsw;

    // Deserialize index and check recall
    // 反序列化，并检查召回率
    // alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    // correct = 0;
    // for (int i = 0; i < max_elements; i++) {
    //     std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
    //     hnswlib::labeltype label = result.top().second;
    //     if (label == i) correct++;
    // }
    // recall = (float)correct / max_elements;
    // std::cout << "Recall of deserialized index: " << recall << "\n";

    // delete[] data;
    // delete alg_hnsw;
    
    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();
    // 计算耗费的时间（以毫秒为单位）
    std::chrono::duration<double, std::milli> elapsed = end - start;
    
    std::cout << "耗时: " << elapsed.count() << " 毫秒" << std::endl;
    return 0;
}
