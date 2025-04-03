// 无论该头文件被引用多少次，都只编译一次，pragma源自于拉丁语，它的含义是“行动”或“做某事”
#pragma once

#include "visited_list_pool.h"  // 引入了mutex
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h> // malloc, free
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>   // 智能指针

namespace hnswlib {
// using tableint = unsigned int;   四个字节，一个tableint存储一个邻居节点的id（id和label不一样）
typedef unsigned int tableint; 
// 四个字节，对于第0层是size(2)+flag+reserved，对于第0层以上的，就是size(2)+reserved
typedef unsigned int linklistsizeint;   // size表示该层邻居节点的数量

template<typename dist_t>   // dist_t一般就是float，hnswlib::HierarchicalNSW<dist_t>* alg_hnsw
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;    // 只有static的话，静态成员变量，类内定义，类外初始化
    static const unsigned char DELETE_MARK = 0x01;  // VBASE中被删除

    // 成员变量可以用大括号初始化，也可以直接使用等号，但不能使用小括号(Effective Modern C++)
    size_t max_elements_{0};    // size_t是unsigned long long
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements，加mutable的原因是，可以在const 成员函数中修改
    size_t size_data_per_element_{0};   // linklistsizeint + neighbors + data + label 第0层的，size_data_per_element_ = 4 + 16 * 2 * 4 + 128 * 4 + 4
    size_t size_links_per_element_{0};  // linklistsizeint + neighbors  1层以上的
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };    // searchBaseLayerST

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};   // 最大层数

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};   // VisitedListPool是visited_list_pool.h中的一个类，在构造函数中初始化
    
    mutable std::vector<std::mutex> label_op_locks_;    // 

    std::mutex global;  // 在addPoint函数中，如果当前加入的这个节点随机到的层数比最大层数还要大，后续的操作都要加锁
    std::vector<std::mutex> link_list_locks_;   // 大小为max_elements，我靠，我感觉好多都用到了这个啊，啥情况

    tableint enterpoint_node_{0};   // 莫非是整个图的入口点

    size_t size_links_level0_{0};   // 第0层的linklistsizeint + neighbors
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    char *data_level0_memory_{nullptr}; // 懂，char，说明是按照字节来算的
    char **linkLists_{nullptr}; // 懂
    std::vector<int> element_levels_;  // element_levels_[i] 的值为 3，则说明索引中第 i 个节点的最高层级是 3

    size_t data_size_{0};   // 128维度 * 4，单位为B

    DISTFUNC<dist_t> fstdistfunc_;  // 函数指针类型
    void *dist_func_param_{nullptr};    // 指针类型

    std::unordered_map<labeltype, tableint> label_lookup_;  // typedef size_t labeltype;
    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    
    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;   // 和updatePoint有关的

    mutable std::atomic<long> metric_distance_computations{0};  // 统计距离计算的总次数
    mutable std::atomic<long> metric_hops{0};   // 节点之间跳跃的次数

    bool allow_replace_deleted_ = false;  // 删除节点后，HNSW 的索引结构中会保留这个被删除的节点位置，新插入的节点将获得新的 ID。在MSVBASE中直接被删除了？

    std::unordered_set<tableint> deleted_elements;  // 只有允许替换已删除的节点时才会用到，但默认是false
    std::mutex deleted_elements_lock;  // lock for deleted_elements
    
    int num_level1;

    HierarchicalNSW(SpaceInterface<dist_t> *s) {
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,    // 这个0代表什么意思？
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        // 调用了hnswalg.h中的一个函数
        loadIndex(location, s, max_elements);
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        bool allow_replace_deleted = false) // 是否允许替换已删除的节点
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),   // 初始化这个数组的大小为65536，且每个元素都是mutex
            link_list_locks_(max_elements), // 初始化数组大小
            element_levels_(max_elements),  // 初始化数组大小，且每个元素都是0
            allow_replace_deleted_(allow_replace_deleted) {
        max_elements_ = max_elements;
        num_level1 = 0;
        num_deleted_ = 0;   // 删除元素计数器初始化为0
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if ( M <= 10000 ) {
            M_ = M;
        } else {
            // #define HNSWERR std::cerr
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_; // 当前节点的最大连接数
        maxM0_ = M_ * 2;    // 第零层一个元素能建立的最大连接数
        ef_construction_ = std::max(ef_construction, M_);   // 动态候选列表的大小
        ef_ = 10;   // ef 的初始值设为 10

        level_generator_.seed(random_seed); // 决定新增的向量在哪一层，由于种子相同，所以每次运行结果一样，tests/new/random_test.cpp中测试了
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);   // 3 + 1
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);   // 计算每个元素的数据大小，包括邻居、数据和标签，3 + 3
        
        offsetData_ = size_links_level0_;   // 设置数据在内存中的偏移量
        label_offset_ = size_links_level0_ + data_size_;    // 标签在内存中的偏移量
        offsetLevel0_ = 0;
        // 为第0层的数据分配内存，存储每个元素的链接和数据，都是按一个字节算大小
        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");  // 需要引入#include <stdexcept>

        cur_element_count = 0;  // 第零层现有元素个数
        // 图操作经常需要判断哪些节点已经走过，这里提供一个已经申请好空间的池子，减少内存频繁申请释放的开销
        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        enterpoint_node_ = -1;  // 表示还没有
        maxlevel_ = -1;
        // sizeof(void *)表示指针类型的大小
        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        // 计算每个元素的链接占用的内存大小（不包括第0层）
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);  //用于计算新增向量落在哪一层
        revSize_ = 1.0 / mult_;
    }


    ~HierarchicalNSW() {
        clear();
    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {  // cur_element_count表示第零层总共有多少个节点
            if (element_levels_[i] > 0) // 等于0 在data_level0_memory_中被释放了
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count = 0;  // 原子类型，相当于cur_element_count.store(0)
        visited_list_pool_.reset(nullptr);  // 好像是unique_ptr的reset函数
    }

    // 想根据 first 元素进行排序，降序
    struct CompareByFirst {
        //constexpr，noexcept c++11新特性
        // 编译器可以在编译时计算 constexpr 表达式，从而减少运行时的开销。
        constexpr bool operator()(std::pair<dist_t, tableint> const& a, // 常量引用，风格和习惯问题
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };

    void setEf(size_t ef) {
        ef_ = ef;
    }

    // 通过哈希方法计算一个 label 对应的mutex，确保在多线程操作中不同的 label 可能会使用不同的互斥量来进行锁定，从而减少锁争用
    inline std::mutex& getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);   // lock_id <= 65535
        return label_op_locks_[lock_id];    // 返回一个mutex
    }
    // 根据internal_id获取label
    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }
    // 给internal_id那个node的label字段赋值，主要用在如果被标记删除的节点可以被替代
    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }

    // 根据id返回node中的label的指针，用在addPoint时初始化该节点的label
    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }

    // 根据id返回node中的data的指针
    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }

    // 获取随机层数，和M有关，M=16的时候，reverse_size为0.83，最多大概是5-7层
    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);  // 均匀分布
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }
    linklistsizeint *get_linklist0(tableint internal_id) const {    // offsetLevel0_初始化为0
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }

    // 没有用到过
    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }

    // 用到了
    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }
    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;    // flag在第3个字节
        return *ll_cur & DELETE_MARK;   // 返回1，也就是true，表示被标记为删除
    }

    // 邻居的数量
    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);    
    }


    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }

    // 没有被使用过，getDataByInternalId用的比较多，一个返回指针，一个返回vector
    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        
        std::unique_lock <std::mutex> lock_table(label_lookup_lock);    // label和table不一样
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*) data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }
    // 加锁获取所有的邻居的id，只用在updatePoint使用
    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);  // 返回linklistsizeint
        int size = getListCount(data);  // 返回邻居数量
        std::vector<tableint> result(size); 
        tableint *ll = (tableint *) (data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }
    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count); // 如果条件为真，程序继续正常运行；如果条件为假，程序会立即终止，并输出错误信息
        if (!isMarkedDeleted(internalId)) {
            // 第3个字节是flag
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {   // 意味着系统允许新插入的节点替换已经删除的节点。
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock); // if结束，自动解锁
                deleted_elements.insert(internalId);    // deleted_elements是一个set集合
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }
    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void markDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));   // unique_lock是类，mutex是模板，函数结束自动解锁

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);    // mutable std::mutex label_lookup_lock，用来保护label_lookup_
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        markDeletedInternal(internalId);
    }
    /*
    * Remove the deleted mark of the node.
    */
    // 用到过
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }
    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    * 
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void unmarkDelete(labeltype label) {    // 没有被用到过
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);    // label_lookup_lock是一个mutex
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }
    // 没有被用到过，本来就是public，这个函数的意思是什么
    size_t getMaxElements() {
        return max_elements_;   // 一百万
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }
    // 搜索第layer层中离目标最近的ef个邻居，主要用在addPoint和deletePoint函数
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;  // 数组大小好像是100万
        vl_type visited_array_tag = vl->curV;   // -1表示被访问了
        // 优先队列底层存储元素的容器vector，降序存储
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;  // 结果集，存的是正数，最大数量为ef_construction_
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;    // 动态候选集，存的是负数，没有大小限制

        dist_t lowerBound;  // 结果集top_cnadidates中的最远距离
        if (!isMarkedDeleted(ep_id)) {  // 如果起始节点 ep_id没有被标记为删除
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);   // 计算当前节点到目标节点的距离，记为dist
            top_candidates.emplace(dist, ep_id);    // 将当前节点插入结果集W中
            lowerBound = dist;  // 更新结果集中的最远距离为dist
            candidateSet.emplace(-dist, ep_id); // 将当前节点插入动态候选集C中
        } else {    // 如果当前节点已经被标记为删除（注意删除的节点不插入结果集，只插入动态候选集）
            lowerBound = std::numeric_limits<dist_t>::max();    // 更新结果集中的最远距离为当前数据类型dist_t的最大值
            candidateSet.emplace(-lowerBound, ep_id);   
        }

        visited_array[ep_id] = visited_array_tag;      // 添加enterpoint到visited_array

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();  // 弹出动态候选集中当前最近的节点，记为curr_el_pair
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {  // 从候选集取出的最小值 > 结果集的最大值距离且。。
                break;  // 直接结束循环（接下来进入算法的最后阶段，即释放访问列表存储空间，算法结束，返回结果集）
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);   // 加锁，保护当前节点的邻居列表，因为B线程可能正在调整邻居列表大小，或正在插入或删除邻居

            int *data;  
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum); // 获取第零层的linklistsizeint
            } else {
                data = (int*)get_linklist(curNodeNum, layer);   // 获取其他层的
            }
            size_t size = getListCount((linklistsizeint*)data); // size为当前节点的邻居数目
            tableint *datal = (tableint *) (data + 1);  // 指针移动四个字节，即跳过header，指向邻居列表
#ifdef USE_SSE
            // 第二个参数表示预取到L1缓存 第一个参数指定需要预取的内存地址
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif
            // 遍历当前节点的邻居
            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                // 满足一个即可
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id); // 即使被被删除了，也需要加入candidateSet
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif
                    // 如果该节点没有被标记为删除
                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)   // 如果结果集W大小超过ef_construction_
                        top_candidates.pop();   // 就把结果集中距离最大的节点删除

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;    // 更新lowerBound
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }


    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(  // 少了layer，这个算法只在level0中搜索，ST可能真的表示Safe Threading的意思，searchKnn函数会调用它
        tableint ep_id,
        const void *data_point,
        size_t ef,  // max(ef_, k)
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        // bool bare_bone_search = （!num_deleted_ && !isIdAllowed）;
        // 如果是 bare_bone_search（即简化搜索模式），那么直接进入 if。如果不是简化搜索模式，ep_id 不能被标记为删除，且没有isIdAllowed 过滤器，如果存在isIdAllowed 过滤器，则外部标签 getExternalLabel(ep_id) 必须通过该过滤器的检查
        // isIdAllowed是一个指向 BaseFilterFunctor 对象的指针，*isIdAllowed 是一个函数对象，它接受一个 labeltype 类型的参数，返回一个 bool 类型的值
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;  // 初始化lowerBound
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {  // 默认不走这里
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {    // ep_id 被标记为删除或者没有通过过滤器
            lowerBound = std::numeric_limits<dist_t>::max();    // candiate_set只可能有一个负无穷，而top_candidates肯定不会有正无穷
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound; // searchBaseLayer还需要加上&& top_candidates.size() == ef_construction_
            } else {    // 如果不是简化模式
                if (stop_condition) {   // BaseSearchStopCondition<dist_t>* stop_condition = nullptr
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {    // 不是简化模式就默认走这
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }   // 这13行就一个任务，判断是否要结束搜索
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id); // 邻居节点data
            size_t size = getListCount((linklistsizeint*)data); // 邻居节点数量
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {  // false，我觉得开启也挺好的
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {    // j从1开始是因为，data指针没有指向neighbors的第一个元素
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {  // 如果没有访问
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {    // 简化模式或者没有停止条件
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;  // 满足一个条件继续往下走，不满足就直接判断下一个邻居。searchBaseLayer也是这个逻辑
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id); // 在判断该邻居节点时，如果结果集>=50且该邻居节点大于top的最大值
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {  // 默认不走这
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {  // 不为简化模式且有停止条件
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    // Heuristic启发式，避免孤岛效应
    // 输入参数top_candidates会被清空，然后选择合适的点重新装进去
    void getNeighborsByHeuristic2(  
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest; // 默认降序
        std::vector<std::pair<dist_t, tableint>> return_list;   
        while (top_candidates.size() > 0) { // 初始化queue_closest
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {  // 每选择一个节点时，都要看此节点到目标节点a的距离是否比此节点到所有已选中的节点远，如果远，则选中它
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();  // 负的
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                        getDataByInternalId(curent_pair.second),
                                        dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second); // 正的
        }
    }

    // addPoint和 updatePoint调用repairConnectionsForUpdate函数，repairConnectionsForUpdate调用它
    // 让新元素与其他节点“相互”建立连接，返回值是距离目标节点最近的邻居的tableint
    tableint mutuallyConnectNewElement( 
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);   // 为什么不是Mcurmax？
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);  // 预留指定数量的空间
        while (top_candidates.size() > 0) { // 把已经启发式之后的top_candidates的内部id，全部放入放入selectedNeighbors
            selectedNeighbors.push_back(top_candidates.top().second);   // 把top_candidates的id放入selectedNeighbors
            top_candidates.pop();   // 清空top_candidates
        }

        tableint next_closest_entry_point = selectedNeighbors.back();   // 返回最后一个元素，距离目标节点最近的，好像只在return那里用到了

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);   // defer_lock，保证在构造时，保持未锁定
            if (isUpdate) {
                lock.lock();    // 更新才需要加锁
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);  // 得到一个linklistsizeint
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) { // 如果是新增，且ll_cur中还有值
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size()); // 不管是不是level0，size都是最高的两个字节
            tableint *data = (tableint *) (ll_cur + 1); // 跳过4个字节，直接找到neighbors
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx]; // 插入到neighbors区域（大小固定）
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);   // 获取邻居的linklistsizeint
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other); // 获取当前遍历到的邻居的邻居数量

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);   // 获取当前邻居的第一个邻居指针

            bool is_cur_c_present = false;  // 判断当前遍历到的这个邻居，它在更新之前，是否已经和要更新的节点是邻居了
            if (isUpdate) { // 如果是更新操作
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;    // 相当于continue，再执行for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)的下一个
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {    // 当前遍历到的这个邻居，它在更新之前，和目标节点不是邻居，则需要进行如下操作
                if (sz_link_list_other < Mcurmax) { // 如果当前遍历到的这个邻居可连接的数量还够用，那就不用管了，直接“相互”连接
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {    // finding the "weakest" element to replace it with the new one
                    
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),    // 先计算一下节点和当前遍历的邻居的距离
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {   // 把当前遍历到的这个邻居的所有邻居也加入这个candidates
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);  // 进行一次启发式搜索，得到的candidates的结果从33变成了19

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;   
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }

        return next_closest_entry_point;
    }

    void repairConnectionsForUpdate(
        const void *dataPoint,
        tableint entryPointInternalId,
        tableint dataPointInternalId,
        int dataPointLevel,
        int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                        tableint cand = datal[i];
                        dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level); 

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }
                // 连接节点与邻居
                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }
    // 即使你真的想更新，也需要先把它当成插入，如果发现原先就有那才进行跟新，所以updatePoint只在addPoint中调用
    // 调用的时候updateNeighborProbability参数都是1.0
    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
        // 修改level0中的data数据
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];    // 获取要更新的节点所在的最高层
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        // 从每第0层开始，到该节点的最高层    
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;    // jeven: store part of original neighbors need to be updated，在当前层中确定需要更新的部分邻居。是否更新某邻居由随机数与updateNeighborProbability的比较决定。
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);   // jeven: 获取原始的邻居，当前节点的第一跳邻居
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)    // 都要更新，不会走continue
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop); // sCand存了自己和两圈的邻居
                }
            }

            for (auto&& neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;  // 从大到小
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed to have size >= 1，正常情况下都是sCand.size() - 1
                size_t elementsToKeep = std::min(ef_construction_, size);   // 不能大于ef_construction大小
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }   // 获得sNeigh的每一个元素的两跳内的candidates

                // Retrieve neighbours using heuristic and set connections.
                getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                {
                    std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);   // 获取邻居列表的起始位置
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *) (ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;    // 更新linkLists_或data_level0_memory_
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy); // 相当于从elemLevel层开始到最底层，添加了一个新节点似的
    }

    tableint addPoint(const void *data_point, labeltype label, int level) {
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {    // 如果存在的话，直接更新节点
                tableint existingInternalId = search->second;   // 得到label对应的内部节点id
                if (allow_replace_deleted_) {   // 
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");   
                    }
                }
                lock_table.unlock();
                // 走到这个if，就是单纯的更新data，不更新label和id
                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                updatePoint(data_point, existingInternalId, 1.0);   

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }
            // 能走到这一步，说明是新增节点
            cur_c = cur_element_count;  // jeven: 1. 节点id自增，为cur_c
            cur_element_count++;    // 相当于cur_element_count.fetch_add(1)
            label_lookup_[label] = cur_c;
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);     // 可能有的线程在读

        int curlevel = getRandomLevel(mult_);   // jeven: 2.随机初始化层数 curlevel，第0层的概率0.9375，第1层的概率0.0586，第二层0.0002
        if (level > 0)  
            curlevel = level;   // 如果level = 0，说明是layer0，那此时我们就随机产生一个curlevel，作为这个节点的最高层次。但好像下面的addpoint传递的是-1

        element_levels_[cur_c] = curlevel;

        std::unique_lock <std::mutex> templock(global); 
        int maxlevelcopy = maxlevel_;   // maxlevel_成员变量，初始值为-1
        if (curlevel <= maxlevelcopy)
            templock.unlock();  // 意义不明的加锁和解锁
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;
        // jeven: 3. 初始化节点相关数据结构，把label和data的内容填好
        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);    // 0表示要填入的内容，第3个参数表示要设置的字节数，就是把整个node0都初始化为0

        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));    // 参数一，目标内存块指针，参数三，要复制的字节数
        memcpy(getDataByInternalId(cur_c), data_point, data_size_); 
        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);    // 初始化非零层的数据结构
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);   // level1到curlevel全部设置为0
        }
        
        // jeven: 5.待添加的节点不是第一个元素
        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {  // 这个 if目的是找到curlevel层的
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) { // jeven: 5.1 逐层往下寻找直至curlevel+1，找到最近的节点
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);  // 这个锁挺关键的，因为获取datal、data和size时，有可能其他线程正在修改currObj的邻居
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }   // 结束后到达curlevel

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {   // jeven: 5.2 从curlevel往下，找一定数量的邻居并连接
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);    // top_candidates.size() == ef_construction
                if (epDeleted) {    // 把enterpoint_copy加入top_candidates，目前这个用意还不清楚
                    top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                // jeven: 5.3 连接节点与邻居
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            // jeven: 4. 如果是第一个元素，只需更新起始点和最大层
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {  // 只有第0层时，enterpoint就是0，当出现第一层时，enterpoint会一直为第一层第一个出现的，除非第二层出现
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;   // 返回这个新插入节点的id
    }

    /*
    * Adds point. Updates the point if it is already in the index.
    * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
    */
    // 真正调用，alg_hnsw->addPoint(point_data, label);
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));   // return label_op_locks_[lock_id];
        if (!replace_deleted) {    // 如果不需要替换删除节点
            addPoint(data_point, label, -1);    // -1是level的值
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock); // 加锁，markDeletedInternal函数也会操作deleted_elements这个set
        bool is_vacant_place = !deleted_elements.empty();   // 如果有被删除的元素，deleted_elements.empty()为false，is_vacant_place为真
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();   // deleted_elements是unordered_set，begin()取得是指针
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();     // 解锁

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addPoint(data_point, label, -1);    // 即使没有空闲的位置的情况，也可以更新updatePoint
        } else {    // 有空闲的位置
            // we assume that there are no concurrent operations on deleted element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);  // 主要是为了更新label->id的映射表
            setExternalLabel(internal_id_replaced, label);  // 把label放到数据结构中

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;    // 更新label_lookup表
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0); 
        }
    }

    std::priority_queue<std::pair<dist_t, labeltype >>  // 距离按从大到小排序
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;    // enterpoint_node_在最高层
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        // 0层以上的搜索逻辑
        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;    // 在每一层，先找一圈邻居中，距离目标节点最近的邻居，再根据该邻居又找一圈，贪心到没有更近的为止
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);   // 上一层有的点，下一层一定会有
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);  // neighbors的起始地址
                for (int i = 0; i < size; i++) {    
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }   // 结束之后，得到第0层的currObj

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;  // 距离从大到小
        // isIdAllowed 为nullptr，!isIdAllowed 的值将为 true，只要没有没有删除的元素，即为true，目前就认为是true
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(   // 不是很理解为什么不开启collect_metrics
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        }

        while (top_candidates.size() > k) { // top_candidates的大小可能大于ef
            top_candidates.pop();
        }
        // id转换为label
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    std::vector<std::pair<dist_t, labeltype >>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::vector<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }

    // 验证图的结构完整性的调试工具或验证工具
    void checkIntegrity() {
        int connections_checked = 0;
        std::vector <int> inbound_connections_num(cur_element_count, 0);    // 所有的都初始化为0
        for (int i = 0; i < cur_element_count; i++) {   // 对于每一个节点
            for (int l = 0; l <= element_levels_[i]; l++) { // 它的每一层
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);    // 获取邻居节点的数量
                tableint *data = (tableint *) (ll_cur + 1); // 获取邻居节点的指针
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);   // 确保邻居没有重复值，因为s是set
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);  // 统计所有节点中入边最小的为多少
                max1 = std::max(inbound_connections_num[i], max1);  
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }
    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);  // hnswlib.h，POD表示一些简单类型数据
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);
        // level0的数据
        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);
        // levels 大于0的数据
        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }
        output.close();
    }
    // 在构造函数中调用了loadIndex(location, s, max_elements);其中max_elements默认为0
    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        clear();    // 析构函数调用的那个函数
        // get file size:
        input.seekg(0, input.end);  // 第二个参数是相对位置，第一个参数是偏移量
        std::streampos total_filesize = input.tellg();  // tellg()返回当前文件指针的位置，类型为 std::streampos，单位为字节
        input.seekg(0, input.beg);  // 将文件指针移动到文件开头

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;   // 4行的意思就是说，如果要更新max_elements_这个成员变量，你就通过参数传进来，否则就用默认值
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        /// Optional - check if index is ok:
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }
        }

        // throw exception if it either corrupted or old index
        if (input.tellg() != total_filesize)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();  // 用于清除流的错误状态标志，使流恢复到正常状态，如果流中没有错误发生，clear()函数不会做任何事情
        /// Optional check end

        input.seekg(pos, input.beg);    // 把指针偏移到level0的数据开始的地方

        data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);   // 第二个构造函数缺少这个初始化成员变量的语句，需要补上

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);   // 同样需要补上
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);   // 创建一个临时的 std::vector<std::mutex>，并将其与 link_list_locks_ 交换
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);   // 可以仿照上面swap的写法
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize); // 读取linkListSize，写入的时候写入的一个结果
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i); // 已经被标记为删除但还没有新节点填充的节点的集合
            }
        }

        input.close();

        return;
    }

    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
    }

    size_t indexFileSize() const {
        size_t size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * size_data_per_element_;

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

};
}  // namespace hnswlib
