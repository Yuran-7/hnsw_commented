#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/space_l2.h"

using namespace std;

// 设定 fvecs 文件路径
const string FVECS_FILE = "/data/ysh/sift/sift_base.fvecs";

// 设定向量维度（如果确定数据集的维度，可直接填入）
const size_t DIM = 128;  // 这里假设是 128 维

// 读取 fvecs 文件
vector<vector<float>> read_fvecs(const string &filename) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "无法打开文件: " << filename << endl;
        exit(1);
    }

    vector<vector<float>> data;
    while (file) {
        int dim;
        file.read((char*)&dim, sizeof(int));  // 读取维度
        if (file.eof()) break;

        if (dim != DIM) {
            cerr << "错误: 文件中的向量维度 (" << dim << ") 与设定 (" << DIM << ") 不匹配！" << endl;
            exit(1);
        }

        vector<float> vec(DIM);
        file.read((char*)vec.data(), DIM * sizeof(float));  // 读取向量
        data.push_back(vec);
    }

    file.close();
    return data;
}

int main() {
    hnswlib::L2Space l2space(DIM);
    vector<vector<float>> vectors = read_fvecs(FVECS_FILE);

    if (vectors.empty()) {
        cerr << "文件读取失败或为空！" << endl;
        return 1;
    }

    cout << "成功读取 " << vectors.size() << " 个向量，每个 " << DIM << " 维" << endl;

    const float *query = vectors[0].data();  // 选择第一个向量作为查询

    auto start = chrono::high_resolution_clock::now();

    for (size_t i = 1; i < vectors.size(); i++) {
        float dist = hnswlib::L2Sqr(query, vectors[i].data(), l2space.get_dist_func_param());
        // cout << "第 " << i << " 个向量的 L2 距离: " << dist << "\r";
        
    }

    auto end = chrono::high_resolution_clock::now();
    double duration = chrono::duration<double>(end - start).count();

    cout << "普通的耗时:" << duration << " 秒" << endl;

    start = chrono::high_resolution_clock::now();

    for (size_t i = 1; i < vectors.size(); i++) {
        float dist = hnswlib::L2SqrSIMD16Ext(query, vectors[i].data(), l2space.get_dist_func_param());
        // cout << "第 " << i << " 个向量的 L2 距离: " << dist << "\r";
    }

    end = chrono::high_resolution_clock::now();
    duration = chrono::duration<double>(end - start).count();

    cout << "SSE的耗时:" << duration << " 秒" << endl;



    start = chrono::high_resolution_clock::now();
    for (size_t i = 1; i < vectors.size(); i++) {
        float dist = hnswlib::L2SqrSIMD16ExtAVX(query, vectors[i].data(), l2space.get_dist_func_param());
        // cout << "第 " << i << " 个向量的 L2 距离: " << dist << "\r";
    }

    end = chrono::high_resolution_clock::now();
    duration = chrono::duration<double>(end - start).count();

    cout << "AVX的耗时:" << duration << " 秒" << endl;
    return 0;
}
