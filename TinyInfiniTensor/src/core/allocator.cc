#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {   
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);
        this -> used += size;
        // Check if there are any free blocks that can be reused
        for (auto it = free_blk.begin(); it != free_blk.end(); ++it)
        {
            if (it->second >= size)
            {
                // Found a free block that can be reused
                size_t offset = it -> first;
                if (it->second > size)
                {
                    // Split the free block if it's larger than needed
                    free_blk[offset + size] = it -> second - size;
                }
                free_blk.erase(it);
                // Do not need to update peak here
                return offset;
            }
            // if it is the last one
            if (it == --free_blk.end()) {
                size_t offset = it -> first;
                // simply extend the last one to the size you want and use it!
                free_blk.erase(it);
                return offset;
            }
        }
        // No free block found, allocate a new block
        this->peak += size;
        return this->peak - size;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);
        this->used -= size;
        // Check if the free block can be merged with adjacent free blocks
        for (auto it = free_blk.begin(); it != free_blk.end(); ++it) {
            if (it -> first + it -> second == addr) {
                free_blk[it -> first] += size;
                return;
            }
            else if (addr + size == it -> first) {
                free_blk[addr] = size + it -> second;
                free_blk.erase(it);
                return;
            }
        }
        // Else we add a new free blk
        free_blk[addr] = size;
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        // round up to the multiple of alignment
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
