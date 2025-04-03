#pragma once
// D:\JDK\mingw64\lib\gcc\x86_64-w64-mingw32\8.1.0\include\c++
#include <unordered_map>
#include <fstream>
#include <mutex>
// c++标准库头文件是没有.h或.hpp的
#include <algorithm>
#include <assert.h>

namespace hnswlib {
template<typename dist_t>   // 在test中，typedef float dist_t
class BruteforceSearch : public AlgorithmInterface<dist_t> {    // 在epsilon_search_test.cpp中用到了
 public:
    char *data_;
    size_t maxelements_;
    size_t cur_element_count;
    size_t size_per_element_;

    size_t data_size_;
    DISTFUNC <dist_t> fstdistfunc_;
    void *dist_func_param_;
    std::mutex index_lock;

    std::unordered_map<labeltype, size_t > dict_external_to_internal;


    BruteforceSearch(SpaceInterface <dist_t> *s)
        : data_(nullptr),
            maxelements_(0),
            cur_element_count(0),
            size_per_element_(0),
            data_size_(0),
            dist_func_param_(nullptr) {
    }


    BruteforceSearch(SpaceInterface<dist_t> *s, const std::string &location)
        : data_(nullptr),
            maxelements_(0),
            cur_element_count(0),
            size_per_element_(0),
            data_size_(0),
            dist_func_param_(nullptr) {
        loadIndex(location, s);
    }


    BruteforceSearch(SpaceInterface <dist_t> *s, size_t maxElements) {
        maxelements_ = maxElements;
        data_size_ = s->get_data_size();    // dim * sizeof(float)
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        size_per_element_ = data_size_ + sizeof(labeltype);
        data_ = (char *) malloc(maxElements * size_per_element_);
        if (data_ == nullptr)
            throw std::runtime_error("Not enough memory: BruteforceSearch failed to allocate data");
        cur_element_count = 0;
    }


    ~BruteforceSearch() {
        free(data_);
    }

    // 这个函数是有用的，计算召回率也需要alg_brute.addPoint(point_data, label)
    void addPoint(const void *datapoint, labeltype label, bool replace_deleted = false) {
        int idx;
        {
            std::unique_lock<std::mutex> lock(index_lock);  // 加锁

            auto search = dict_external_to_internal.find(label);
            // 如果标签存在
            if (search != dict_external_to_internal.end()) {
                idx = search->second;
            } else {
                if (cur_element_count >= maxelements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit\n");
                }
                // 分配新索引
                idx = cur_element_count;    // cur_element_count，类的成员变量
                dict_external_to_internal[label] = idx; // 字典， 外部的标签映射到内部的索引
                cur_element_count++;
            }
        }
        // 将标签信息存储到相应的内存位置
        memcpy(data_ + size_per_element_ * idx + data_size_, &label, sizeof(labeltype));
        // 将实际数据点存储到相应的内存位置
        memcpy(data_ + size_per_element_ * idx, datapoint, data_size_);
    }


    void removePoint(labeltype cur_external) {
        std::unique_lock<std::mutex> lock(index_lock);

        auto found = dict_external_to_internal.find(cur_external);
        if (found == dict_external_to_internal.end()) {
            return;
        }

        dict_external_to_internal.erase(found);

        size_t cur_c = found->second;   // cur_c和idx的效果差不多
        labeltype label = *((labeltype*)(data_ + size_per_element_ * (cur_element_count-1) + data_size_));
        dict_external_to_internal[label] = cur_c;
        memcpy(data_ + size_per_element_ * cur_c,
                data_ + size_per_element_ * (cur_element_count-1),
                data_size_+sizeof(labeltype));
        cur_element_count--;
    }

    // 返回值
    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        // 断言检查，确保 k 不大于当前已有的元素数
        assert(k <= cur_element_count);
        // 按距离降序
        std::priority_queue<std::pair<dist_t, labeltype >> topResults;
        if (cur_element_count == 0) 
            return topResults;
        // 遍历前 k 个元素，计算它们到查询点的距离，并初始化优先队列
        for (int i = 0; i < k; i++) {
            dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
            labeltype label = *((labeltype*) (data_ + size_per_element_ * i + data_size_));
            // 如果没有过滤器或标签通过过滤器，则将距离和标签加入优先队列
            if ((!isIdAllowed) || (*isIdAllowed)(label)) {
                topResults.emplace(dist, label);
            }
        }
        dist_t lastdist = topResults.empty() ? std::numeric_limits<dist_t>::max() : topResults.top().first;
        for (int i = k; i < cur_element_count; i++) {
            dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
            // 如果当前数据点距离小于或等于优先队列中最远的点，则有可能加入队列
            if (dist <= lastdist) {
                labeltype label = *((labeltype *) (data_ + size_per_element_ * i + data_size_));
                // 替换
                if ((!isIdAllowed) || (*isIdAllowed)(label)) {
                    topResults.emplace(dist, label);
                }
                if (topResults.size() > k)
                    topResults.pop();
                // 更新lastdist
                if (!topResults.empty()) {
                    lastdist = topResults.top().first;
                }
            }
        }
        return topResults;
    }


    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, maxelements_);
        writeBinaryPOD(output, size_per_element_);
        writeBinaryPOD(output, cur_element_count);

        output.write(data_, maxelements_ * size_per_element_);

        output.close();
    }


    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s) {
        std::ifstream input(location, std::ios::binary);
        std::streampos position;

        readBinaryPOD(input, maxelements_);
        readBinaryPOD(input, size_per_element_);
        readBinaryPOD(input, cur_element_count);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        size_per_element_ = data_size_ + sizeof(labeltype);
        data_ = (char *) malloc(maxelements_ * size_per_element_);
        if (data_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate data");

        input.read(data_, maxelements_ * size_per_element_);

        input.close();
    }
};
}  // namespace hnswlib
