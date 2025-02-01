#include "core/graph.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        // hash table
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    // void GraphObj::optimize()
    // {
    //     // =================================== 作业 ===================================
    //     // TODO: 设计一个算法来实现指定的图优化规则
    //     // 图优化规则如下：
    //     // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）如何判断相邻？前驱后继关系 如何删除？removeOp/removeTensor
    //     // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
    //     // =================================== 作业 ===================================
    //     // 拓扑排序
    //     IT_ASSERT(topo_sort() == true);
    //     std::cout << "before optimize" << std::endl;
    //     // store the ops need to be removed
    //     std::unordered_set<Operator> removeOps;
    //     std::unordered_set<Tensor> removeTensors;
        
    //     // ops has been sorted
    //     for (auto &op: ops) {
    //         std::cout << "op: " << std::endl;
    //         std::cout << op -> toString() << std::endl;
    //         auto prevOps = op -> getPredecessors();
    //         std::cout << "prevOps: " << std::endl;
    //         for (auto prevOp: prevOps) { std::cout << prevOp -> toString() << std::endl; }


    //         // consider 2 transpose ops
    //         if (prevOps.size() != 0 &&
    //             op-> getOpType() == infini::OpType::Transpose &&
    //             prevOps[0] -> getOpType() == infini::OpType::Transpose) 
    //         {
    //             std::cout << "merge 2 transposes" << std::endl;
    //             auto s1 = dynamic_cast<TransposeObj*>(prevOps[0].get()) -> getPermute(), s2 = dynamic_cast<TransposeObj*>(op.get()) -> getPermute();
    //             Tensor tensor0 = prevOps[0] -> getInputs()[0], tensor1 = op -> getInputs()[0], tensor2 = op -> getOutput();
    //             Operator op1 = tensor0 -> getSource();


    //             // if s1 == s2, remove transpose ops
    //             if (s1 == s2) {
    //                 // renew tensor0 attributes
    //                 std::cout << "renew tensor0 attributes" << std::endl;
    //                 tensor0 -> removeTarget(prevOps[0]);
    //                 for (auto &target: tensor2 -> getTargets()) {
    //                     tensor0 -> addTarget(target);
    //                 }


    //                 // renew op2 attributes
    //                 std::cout << "renew op2 attributes" << std::endl;
    //                 for (auto op2: tensor2 -> getTargets()) {
    //                     op2 -> removePredecessors(op);
    //                     op2 -> addPredecessors(op1);
    //                 }


    //                 // renew op1 attributes
    //                 std::cout << "renew op1 attributes" << std::endl;
    //                 for (auto &op: ops) { std::cout << op -> toString() << std::endl; }
    //                 for (auto &target: tensor0 -> getTargets()) {
    //                     target -> replaceInput(tensor2, tensor0);
    //                     target -> removePredecessors(op);
    //                     target -> addPredecessors(prevOps[0]);
    //                 }


    //                 // remove useless tensors
    //                 std::cout << "remove useless tensors" << std::endl;
    //                 removeTensors.insert(tensor1);
    //                 removeTensors.insert(tensor2);


    //                 // remove useless ops
    //                 std::cout << "remove useless ops" << std::endl;
    //                 removeOps.insert(op);
    //                 removeOps.insert(prevOps[0]);

    //             } 
    //             // if s1 != s2, merge transpose ops
    //             // else { 
                    
    //             // }
                
    //         }
    //         // merge transpose and matmul 
    //         // else if(op -> getOpType() == infini::OpType::MatMul) {
    //         //     auto &inputs = op -> getInputs();
    //         //     for (auto prevOp: prevOps) {
    //         //         if (prevOp -> getOpType() == infini::OpType::Transpose) {
    //         //             auto it = std::find(inputs.begin(), inputs.end(), prevOp -> getOutput());
    //         //             if (it - inputs.begin() == 0) { }
    //         //         }
    //         //     }
    //         // }
    //     }
    //     std::cout << "after optimize" << std::endl;
    //     // for (auto &op: removeOps) { removeOperator(op); }
    //     // for (auto &tensor: removeTensors) { removeTensor(tensor); }
    // }

    void GraphObj::optimize() {
  using namespace infini;

  // Keep looping until no more optimization can be done
  bool optimized = true;
  while (optimized) {
    optimized = false;
    for (auto it = ops.begin(); it != ops.end(); ++it) {
      const auto op = *it;

      if (op->type == OpType::Transpose) {
        // 1. Two consecutive transpose operators should be simplified
        auto next = std::next(it);
        if (next != ops.end() && next->get()->type == OpType::Transpose) {
          auto next_op = *next;
          auto perm1 = as<TransposeObj>(op)->getPermute();
          auto perm2 = as<TransposeObj>(next_op)->getPermute();
          auto perm = perm1;
          for (size_t i = 0; i < perm.size(); ++i) {
            perm[i] = perm2[perm1[i]];
          }
          auto naive = std::all_of(perm.begin(), perm.end(),
                                   [i = 0](int p) mutable { return p == i++; });

          if (naive) {
            // Delete both
            op->getInputs(0)->removeTarget(op);
            for (auto succ : next_op->getSuccessors()) {
              op->getInputs(0)->addTarget(succ);
              succ->removePredecessors(next_op);
              succ->replaceInput(next_op->getOutput(), op->getInputs(0));
            }

            for (auto pred : op->getPredecessors()) {
              next_op->getOutputs()[0]->setSource(op);
              pred->removeSuccessors(op);
            }

            if (next_op->getSuccessors().size() != 0 &&
                op->getPredecessors().size() != 0) {
              for (auto succ : next_op->getSuccessors()) {
                for (auto pred : op->getPredecessors()) {
                  pred->addSuccessors(succ);
                  succ->addPredecessors(pred);
                }
              }
            }

            removeTensor(next_op->getOutput());
            removeTensor(op->getOutput());
            removeOperator(next_op);
            removeOperator(op);
          } else {
            auto newOp = addOpWithOutputs<TransposeObj>(
                op->getInputs(0), next_op->getOutput(), perm);

            op->getInputs(0)->removeTarget(op);
            for (auto pred : op->getPredecessors()) {
              pred->removeSuccessors(op);
            }

            for (auto succ : next_op->getSuccessors()) {
              succ->removePredecessors(next_op);
            }
            removeTensor(op->getOutput());
            removeOperator(op);
            removeOperator(next_op);
          }

          optimized = true;
          break;
        }
      }

      if (op->type == OpType::MatMul) {
        // 2. Merge transpose operators into matmul operators
        const auto &matmul_op = as<MatmulObj>(op);
        const auto &inputs = matmul_op->getInputs();
        // check if inputs have a transpose operator for last two
        // dimensions
        for (const auto &input : inputs) {
          if (input->getSource() == nullptr ||
              input->getSource()->type != OpType::Transpose)
            continue;
          const auto &transpose_op = as<TransposeObj>(input->getSource());

          // Other dimensions should not be transposed
          auto perm = transpose_op->getPermute();
          if (!std::all_of(perm.begin(), perm.begin() + perm.size() - 2,
                           [i = 0](int p) mutable { return p == i++; }))
            continue;

          // Last two dimensions should be transposed
          if (perm[perm.size() - 1] - perm[perm.size() - 2] == 1)
            continue;

          // Merge transpose into matmul
          if (input == inputs[0]) {
            matmul_op->setTransA(!matmul_op->getTransA());
          } else {
            matmul_op->setTransB(!matmul_op->getTransB());
          }

          // Connect the matmul with the input of the transpose operator
          matmul_op->replaceInput(input, transpose_op->getInputs(0));
          transpose_op->getInputs(0)->removeTarget(transpose_op);
          transpose_op->getInputs(0)->addTarget(matmul_op);

          // Op connections
          matmul_op->removePredecessors(transpose_op);
          for (auto pred : transpose_op->getPredecessors()) {
            matmul_op->addPredecessors(pred);
            pred->addSuccessors(matmul_op);
            pred->removeSuccessors(transpose_op);
          }

          // Delete the transpose operator
          removeTensor(transpose_op->getOutputs()[0]);
          removeOperator(transpose_op);

          optimized = true;
        }

        if (optimized)
          break;
      }
      // Finish the optimization
    }
  }
}
    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);
        size_t n = tensors.size();
        vector<size_t> offsets(tensors.size(), 0);
        for (size_t i = 0; i < n; ++i) { offsets[i] = allocator.alloc(tensors[i]->getBytes()); }
        void* saddr = allocator.getPtr();
        for (size_t i = 0; i < n; ++i) { tensors[i]->setDataBlob(make_ref<BlobObj>(runtime, reinterpret_cast<char*>(saddr) + offsets[i])); }
        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        
        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini