#include "sampler.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace qwen;

int main() {
    try {
        // --- 1. Greedy Sampling Test ---
        const float logits[] = {1.f, 2.f, 3.f};
        SamplingConfig greedy;
        Sampler a(greedy);

        if (a.sample(logits, 3, {}) != 2) {
            throw std::runtime_error("greedy sampling failed");
        }

        // --- 2. Repetition Penalty Test ---
        const float repeat_logits[] = {3.f, 2.f};
        SamplingConfig repeat;
        repeat.repetition_penalty = 2.f;
        Sampler b(repeat);

        // Logit 3.f for index 0 is penalized (divided by 2.f) -> 1.5f,
        // which makes index 1 with 2.f the greedy winner.
        if (b.sample(repeat_logits, 2, {0}) != 1) {
            throw std::runtime_error("repetition penalty failed");
        }

        // --- 3. Stochastic Sampling Test (Top-K, Top-P, Seeded) ---
        SamplingConfig stochastic;
        stochastic.temperature = 0.8f;
        stochastic.top_k       = 2;
        stochastic.top_p       = 0.9f;
        stochastic.seed        = 123;

        Sampler c(stochastic);
        Sampler d(stochastic);
        std::vector<int> history;

        for (int i = 0; i < 32; ++i) {
            int x = c.sample(logits, 3, history);
            int y = d.sample(logits, 3, history);

            // With logits {1, 2, 3}, index 0 (logit 1) will be excluded by top-k=2
            // Since they use the same seed, x should exactly equal y.
            if (x != y || x == 0) {
                throw std::runtime_error("seeded top-k/top-p sampling failed");
            }
            history.push_back(x);
        }

        std::cout << "PASS: greedy, repetition penalty, top-k/top-p, and seeded sampling\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
