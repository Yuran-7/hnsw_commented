#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <queue>
#include <omp.h>
#include "../../hnswlib/hnswlib.h"

using namespace std;
using namespace hnswlib;

// 定义枚举类型
enum class Dataset
{
    SIFT,
    GIST,
    GLOVE,
    AUDIO,
    MSONG,
    ENRON
};

// 获取 base.fvecs 地址
std::string getBaseAddress(Dataset dataset)
{
    switch (dataset)
    {
    case Dataset::SIFT:
        return "/home/huanzhu/ANNS/dataset/sift/sift_base.fvecs";
    case Dataset::GIST:
        return "D:\\cppProject\\dataset\\gist\\gist_base.fvecs";
    case Dataset::GLOVE:
        return "D:\\cppProject\\dataset\\glove\\glove_base.fvecs";
    case Dataset::AUDIO:
        return "D:\\cppProject\\dataset\\audio\\audio_base.fvecs";
    case Dataset::MSONG:
        return "D:\\cppProject\\dataset\\msong\\msong_base.fvecs";
    case Dataset::ENRON:
        return "D:\\cppProject\\dataset\\enron\\enron_base.fvecs";
    default:
        return "";
    }
}

// 获取 query.fvecs 地址
std::string getQueryAddress(Dataset dataset)
{
    switch (dataset)
    {
    case Dataset::SIFT:
        return "/home/huanzhu/ANNS/dataset/sift/sift_query.fvecs";
    case Dataset::GIST:
        return "D:\\cppProject\\dataset\\gist\\gist_query.fvecs";
    case Dataset::GLOVE:
        return "D:\\cppProject\\dataset\\glove\\glove_query.fvecs";
    case Dataset::AUDIO:
        return "D:\\cppProject\\dataset\\audio\\audio_query.fvecs";
    case Dataset::MSONG:
        return "D:\\cppProject\\dataset\\msong\\msong_query.fvecs";
    case Dataset::ENRON:
        return "D:\\cppProject\\dataset\\enron\\enron_query.fvecs";
    default:
        return "";
    }
}

// 读取 .fvecs 文件
float *load_fvecs(const std::string &path, int &num, int &dim)
{
    ifstream input(path, ios::binary);
    if (!input.is_open())
    {
        cerr << "Error opening file: " << path << endl;
        exit(1);
    }

    input.read(reinterpret_cast<char *>(&dim), 4);
    input.seekg(0, ios::end);
    size_t file_size = input.tellg();
    num = file_size / (4 + dim * 4);

    float *data = new float[num * dim];
    input.seekg(0, ios::beg);

    for (size_t i = 0; i < num; ++i)
    {
        int d;
        input.read(reinterpret_cast<char *>(&d), 4);
        if (d != dim)
        {
            cerr << "File error: inconsistent dimensions." << endl;
            exit(1);
        }
        input.read(reinterpret_cast<char *>(data + i * dim), dim * 4);
    }

    input.close();
    return data;
}

// 读取 .ivecs 文件 (ground truth)
unsigned *load_ivecs(const std::string &path, int &num, int &dim)
{
    ifstream input(path, ios::binary);
    if (!input.is_open())
    {
        cerr << "Error opening file: " << path << endl;
        exit(1);
    }

    input.read(reinterpret_cast<char *>(&dim), 4);
    input.seekg(0, ios::end);
    size_t file_size = input.tellg();
    num = file_size / (4 + dim * 4);

    unsigned *data = new unsigned[num * dim];
    input.seekg(0, ios::beg);

    for (size_t i = 0; i < num; ++i)
    {
        int d;
        input.read(reinterpret_cast<char *>(&d), 4);
        if (d != dim)
        {
            cerr << "File error: inconsistent dimensions." << endl;
            exit(1);
        }
        input.read(reinterpret_cast<char *>(data + i * dim), dim * 4);
    }

    input.close();
    return data;
}


int main()
{
    size_t M = 16;
    size_t efConstruction = 200;
    size_t efSearch = 20;
    size_t k = 10;
    bool hasSave = true;

    Dataset dataset = Dataset::SIFT;

    // 使用 std::string 存储文件路径
    std::string path_base = getBaseAddress(dataset);
    std::string path_query = getQueryAddress(dataset);
    std::string hnsw_path = "/home/huanzhu/ANNS/hnswlib-master/tests/data/sift_16_200.bin";

    if (path_base.empty() || path_query.empty())
    {
        cerr << "Invalid dataset path!" << endl;
        return -1;
    }

    // 加载数据
    int num_base = 0, dim_base = 0, num_query = 0, dim_query = 0;
    float *base_data = load_fvecs(path_base, num_base, dim_base);
    float *query_data = load_fvecs(path_query, num_query, dim_query);

    cout << "Base dim: " << dim_base << ", Query dim: " << dim_query << endl;
    cout << "Loaded dataset. Base size: " << num_base << ", Query size: " << num_query << endl;

    // 构建索引
    L2Space l2space(dim_base);
    hnswlib::HierarchicalNSW<float> *appr_alg;

    if (hasSave)
    {
        cout << "Loading index from file..." << endl;
        appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, hnsw_path);
    }
    else
    {
        cout << "Building index..." << endl;
        appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, num_base, M, efConstruction);

        auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (size_t i = 0; i < num_base; ++i)
        {
            appr_alg->addPoint(base_data + i * dim_base, i);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Building time: " << elapsed.count() << " ms" << std::endl;
        appr_alg->saveIndex(hnsw_path);
    }

    cout << "start search ..." << endl;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<unsigned>> all_results(10000, std::vector<unsigned>(k));

    appr_alg->setEf(efSearch);
    
    for(int i = 0; i < 10000; i++) {
        std::priority_queue<std::pair<float, labeltype>> result = appr_alg->searchKnn(query_data + i * 128, k);

        // 将优先队列转换为向量（优先队列按降序给出结果）
        std::vector<unsigned> result_labels(k);
        for (int j = k - 1; j >= 0; j--) {
            result_labels[j] = result.top().second;
            result.pop();
        }
        
        all_results[i] = result_labels;
    }

    std::ofstream output("/home/huanzhu/ANNS/hnswlib-master/tests/data/res.ivecs", std::ios::binary);
    if (!output.is_open()) {
        std::cerr << "Error opening output file" << std::endl;
        return -1;
    }

    for (int i = 0; i < 10000; i++) {
        const auto& result_labels = all_results[i];
        int result_size = result_labels.size();
        output.write(reinterpret_cast<char*>(&result_size), 4);
        output.write(reinterpret_cast<const char*>(result_labels.data()), result_size * sizeof(unsigned));
    }

    output.close();
    std::cout << "Results saved to " << std::endl;
    

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "search time: " << elapsed.count() << " ms" << std::endl;

    delete[] base_data;
    delete[] query_data;

    return 0;
}
