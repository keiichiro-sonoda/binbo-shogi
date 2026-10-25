// Task 149: obtain plain NNUE metadata from the same generated header as the engine.
#include "eval/nnue/evaluate_nnue.h"
#include <iostream>

int main() {
    using namespace YaneuraOu::Eval::NNUE;
    std::cout << std::hex << kHashValue << " " << FeatureTransformer::GetHashValue()
              << " " << Network::GetHashValue() << "\n";
    std::cout << "Features=" << FeatureTransformer::GetStructureString()
              << ",Network=" << Network::GetStructureString() << "\n";
}
