#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <unordered_set>

using namespace std;

// 读取 ivecs 文件，返回二维向量
vector<vector<int>> read_ivecs(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        exit(1);
    }

    vector<vector<int>> data;
    while (file.good()) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int)); // 读取向量维度
        if (!file.good()) break;

        vector<int> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int)); // 读取向量数据
        data.push_back(vec);
    }

    file.close();
    return data;
}

// 计算 recall@k
double calculate_recall(const vector<vector<int>>& gt, const vector<vector<int>>& res, int k) {
    if (gt.size() != res.size()) {
        cerr << "Error: Groundtruth and result sizes do not match (" << gt.size() << " vs " << res.size() << ")" << endl;
        exit(1);
    }

    int total_queries = gt.size();
    if (total_queries == 0) {
        cerr << "Error: No queries found" << endl;
        exit(1);
    }

    double total_recall = 0.0;
    for (int i = 0; i < total_queries; ++i) {
        const auto& gt_vec = gt[i];
        const auto& res_vec = res[i];

        // 将 groundtruth 的前 k 个元素放入集合
        unordered_set<int> gt_set(gt_vec.begin(), gt_vec.begin() + min(k, (int)gt_vec.size()));

        // 计算 res 中有多少个元素在 gt_set 中
        int hits = 0;
        for (int j = 0; j < min(k, (int)res_vec.size()); ++j) {
            if (gt_set.count(res_vec[j])) {
                hits++;
            }
        }

        // 计算当前查询的 recall@k
        double recall = static_cast<double>(hits) / min(k, (int)gt_vec.size());
        total_recall += recall;
    }

    // 返回平均 recall@k
    return total_recall / total_queries;
}

int main(int argc, char* argv[]) {
    string gt_file, res_file;
    int k = 10; // 默认 k=10，可以通过参数修改

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-gt") == 0 && i + 1 < argc) {
            gt_file = argv[++i];
        } else if (strcmp(argv[i], "-res") == 0 && i + 1 < argc) {
            res_file = argv[++i];
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            k = atoi(argv[++i]);
        } else {
            cerr << "Unknown argument: " << argv[i] << endl;
            return 1;
        }
    }

    // 检查参数是否完整
    if (gt_file.empty() || res_file.empty()) {
        cerr << "Usage: " << argv[0] << " -gt <groundtruth.ivecs> -res <result.ivecs> [-k <k>]" << endl;
        return 1;
    }

    // 读取文件
    cout << "Reading groundtruth from: " << gt_file << endl;
    auto gt = read_ivecs(gt_file);
    cout << "Reading result from: " << res_file << endl;
    auto res = read_ivecs(res_file);

    // 计算 recall
    cout << "Calculating recall@" << k << "..." << endl;
    double recall = calculate_recall(gt, res, k);

    // 输出结果
    cout << "Recall@" << k << " = " << recall << endl;

    return 0;
}