#pragma once

#include <mutex>
#include <string.h>
#include <deque>

namespace hnswlib {
typedef unsigned short int vl_type;
/*
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;
 */

class VisitedList {
 public:
    vl_type curV;
    vl_type *mass;
    unsigned int numelements;

    VisitedList(int numelements1) {
        curV = -1;
        numelements = numelements1;
        mass = new vl_type[numelements];
    }

    void reset() {
        curV++;
        if (curV == 0) {
            memset(mass, 0, sizeof(vl_type) * numelements);
            curV++;
        }
    }

    ~VisitedList() { delete[] mass; }
};
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
    // 默认privite
    std::deque<VisitedList *> pool; 
    std::mutex poolguard;
    int numelements;

 public:
    // visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements)); 
    // visited_list_pool_.reset(nullptr);
    // visited_list_pool_.reset(new VisitedListPool(1, max_elements));   

    VisitedListPool(int initmaxpools, int numelements1) {
        numelements = numelements1;
        for (int i = 0; i < initmaxpools; i++)
            pool.push_front(new VisitedList(numelements));
    }

    VisitedList *getFreeVisitedList() {
        VisitedList *rez;
        {
            std::unique_lock <std::mutex> lock(poolguard);
            if (pool.size() > 0) {
                rez = pool.front();
                pool.pop_front();
            } else {
                rez = new VisitedList(numelements);
            }
        }
        rez->reset();
        return rez;
    }

    void releaseVisitedList(VisitedList *vl) {
        std::unique_lock <std::mutex> lock(poolguard);
        pool.push_front(vl);
    }

    ~VisitedListPool() {
        while (pool.size()) {
            VisitedList *rez = pool.front();
            pool.pop_front();
            delete rez;
        }
    }
};
}  // namespace hnswlib
