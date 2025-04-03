#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <queue>
#include <omp.h>
#include "../../hnswlib/hnswlib.h"

using namespace std;
using namespace hnswlib;

// 读取 .fvecs 文件
float* load_fvecs(const char* path, int& num, int& dim) {
    ifstream input(path, ios::binary);
    if (!input.is_open()) {
        cerr << "Error opening file: " << path << endl;
        exit(1);
    }

    // 每个向量的第一个四字节是维度信息
    input.read(reinterpret_cast<char*>(&dim), 4);
    input.seekg(0, ios::end);   // 将文件指针移动到文件末尾，从文件头开始，移动0个字节
    size_t file_size = input.tellg();   // 获取文件指针的位置，即文件大小，以字节为单位
    num = file_size / (4 + dim * 4);
    
    float* data = new float[num * dim];
    input.seekg(0, ios::beg);   // 将文件指针移动到文件开头，从文件末尾开始，移动0个字节
    
    for (size_t i = 0; i < num; ++i) {
        int d;
        input.read(reinterpret_cast<char*>(&d), 4);  // 读取维度
        if (d != dim) {
            cerr << "File error: inconsistent dimensions." << endl;
            exit(1);
        }
        input.read(reinterpret_cast<char*>(data + i * dim), dim * 4);
    }

    input.close();
    return data;
}

// 读取 .ivecs 文件 (ground truth)
unsigned* load_ivecs(const char* path, int& num, int& dim) {
    ifstream input(path, ios::binary);
    if (!input.is_open()) {
        cerr << "Error opening file: " << path << endl;
        exit(1);
    }

    input.read(reinterpret_cast<char*>(&dim), 4);
    input.seekg(0, ios::end);
    size_t file_size = input.tellg();
    num = file_size / (4 + dim * 4);
    
    unsigned* data = new unsigned[num * dim];
    input.seekg(0, ios::beg);
    
    for (size_t i = 0; i < num; ++i) {
        int d;
        input.read(reinterpret_cast<char*>(&d), 4);
        if (d != dim) {
            cerr << "File error: inconsistent dimensions." << endl;
            exit(1);
        }
        input.read(reinterpret_cast<char*>(data + i * dim), dim * 4);
    }

    input.close();
    return data;
}

std::vector<int> bitmap;    // 全局变量，存储100w行的 label
std::vector<int> filter;    // 全局变量，存储1w行的 label
int query_index = 0;    // 全局变量，查询向量的索引

void load_bitmap(const char* path, int num) {
    ifstream ifs(path, ios::in);
    if (!ifs.is_open()) {
        cerr << "Error opening file: " << path << endl;
        exit(1);
    }

    std::string line;
    int index = 0; // 行号（索引）
    // 逐行读取文件
    while (std::getline(ifs, line) && index < num) {
        std::istringstream iss(line);
        int label;
        iss >> label; // 读取每行的第一个数字作为 label

        // 将 (index, label) 存入 bitmap
        bitmap.push_back(label);

        index++; // 更新行号
    }
    return ;
}

void load_filter(const char* path) {
    ifstream ifs(path, ios::in);
    if (!ifs.is_open()) {
        cerr << "Error opening file: " << path << endl;
        exit(1);
    }
    // 读取第一行
    int num_lines, num_values_per_line;
    ifs >> num_lines >> num_values_per_line;

    // 预分配内存
    filter.reserve(num_lines);

    // 读取接下来的 num_lines 行数据
    for (int i = 0; i < num_lines; ++i) {
        int value;
        ifs >> value;
        filter.push_back(value);
    }
}

class LabelFilter : public BaseFilterFunctor {
public:
    LabelFilter() {}
    ~LabelFilter() {}

    bool operator()(labeltype label) {
       return bitmap[label] == filter[query_index];  // 返回是否相等
    }
};

float test_filter_interface(hnswlib::HierarchicalNSW<float>* index, float* queries, size_t num_queries, size_t dim, size_t k, size_t efSearch) {
    // 设置搜索的 ef 参数
    index->setEf(efSearch);

    // 打开输出文件
    std::ofstream outfile("D:\\cppProject\\hnswlib-master\\tests\\new\\sift_label_filter.out");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open sift_knn.out for writing" << std::endl;
        return -1;
    }
    // 迭代所有查询
    for (size_t i = 0; i < num_queries; ++i) {
        // 执行 KNN 查询
        LabelFilter filter;
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = index->searchKnn(queries + i * dim, k, &filter);    // 降序
        query_index++;  // 更新查询向量的索引
        std::vector<std::pair<float, hnswlib::labeltype>> results_vec;

        // 将结果转移到 vector 中
        while (!result.empty()) {
            results_vec.push_back(result.top());
            result.pop();
        }

        // 逆序输出，每一行 k 个标签
        for (int j = results_vec.size() - 1; j >= 0; --j) {
            // outfile << "(" << results_vec[j].second + 1 << "," << results_vec[j].first << ")";  // 输出标签
            outfile << results_vec[j].second + 1;
            if (j > 0) {
                outfile << " ";  // 间隔符，最后一个标签后没有空格
            }
        }
        outfile << "\n";  // 每个查询的结果换行
    }
    // 关闭文件
    outfile.close();
    std::cout << "Results saved successfully to sift_knn.out" << std::endl;
    
    return 0;  // 返回成功标志
}

int main() {
    // 参数
    size_t M = 4;
    size_t efConstruction = 20;
    size_t efSearch = 100;
    size_t k = 10;
    bool hasSave = true;

    // 文件路径
    const char* path_sift_base = "D:\\cppProject\\dataset\\sift\\sift_base.fvecs";
    const char* path_sift_query = "D:\\cppProject\\dataset\\sift\\sift_query.fvecs";
    const char* path_sift_base_label = "D:\\cppProject\\dataset\\sift\\sift_label\\label_base_10.txt";
    const char* path_sift_query_label = "D:\\cppProject\\dataset\\sift\\sift_label\\label_query_Q1.txt";
    
    load_bitmap(path_sift_base_label, 1000000);    // 读取查询向量的 label
    load_filter(path_sift_query_label);  // 读取过滤器

    // 加载基向量、查询向量和真实集
    int num_base =0, dim_base=0, num_query=0, dim_query=0, num_gt=0, dim_gt = 0;
    float* base_data = load_fvecs(path_sift_base, num_base, dim_base);
    float* query_data = load_fvecs(path_sift_query, num_query, dim_query);

    cout << "Base dim: " << dim_base << ", Query dim: " << dim_query << ", GT dim: " << dim_gt << endl;
    cout << "Loaded SIFT1M dataset. Base size: " << num_base << ", Query size: " << num_query << ", Dim: " << dim_base << endl;

    // 构建索引
    L2Space l2space(dim_base);
    hnswlib::HierarchicalNSW<float>* appr_alg;
    string hnsw_path = "D:\\cppProject\\MSVBASE\\thirdparty\\hnsw\\examples\\data\\sift_4_20.bin";
    if(hasSave == true) {
        cout << "load index..." << endl;
        // 总共有3个构造函数，这是第二个
        appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, hnsw_path);
    } else {
        cout << "Building index..." << endl;
        appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, num_base, M, efConstruction);
        auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (size_t i = 0; i < num_base; ++i) {
            appr_alg->addPoint(base_data + i * dim_base, i);
        }
        auto end = std::chrono::high_resolution_clock::now();   
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "耗时: " << elapsed.count() << " 毫秒" << std::endl;
        appr_alg->saveIndex(hnsw_path);
    }

    auto start = std::chrono::high_resolution_clock::now();
    test_filter_interface(appr_alg, query_data, num_query, dim_base, k, efSearch);
    auto end = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "耗时: " << elapsed.count() << " 毫秒" << std::endl;
    std::cout << "qps: " << num_query / (elapsed.count() / 1000) << std::endl;

}
