#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto A = inputs[0], B = inputs[1];
        auto dimA = A->getDims(), dimB = B->getDims();
        int rankA = dimA.size(), rankB = dimB.size(), rank = std::max(rankA, rankB);
        if (rankA < 2 || rankB < 2) { return std::nullopt; }
        if (transA) { std::swap(dimA[rankA - 1], dimA[rankA - 2]); }
        if (transB) { std::swap(dimB[rankB - 1], dimB[rankB - 2]); }
        // infer the last two dimension
        if (dimB[rankB - 2] == dimA[rankA - 1]) { m = dimA[rankA - 2]; k = dimA[rankA - 1]; n = dimB[rankB - 1]; }
        else { return std::nullopt; }
        // infer the previous part
        auto dimC = infer_broadcast(dimA, dimB);
        dimC[rank - 2] = m; dimC[rank - 1] = n;
        return {{dimC}};
    }

} // namespace infini