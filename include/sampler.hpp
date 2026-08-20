#ifndef QWEN_SAMPLER_HPP
#define QWEN_SAMPLER_HPP

#include <cstdint>
#include <random>
#include <vector>

namespace qwen {

struct SamplingConfig {
    float temperature = 0.0f;
    int top_k = 40;
    float top_p = 0.95f;
    float repetition_penalty = 1.0f;
    int repetition_window = 64;
    uint64_t seed = 0;
    void validate() const;
};

class Sampler {
public:
    explicit Sampler(const SamplingConfig& config);
    int sample(const float* logits, int vocab_size, const std::vector<int>& history);
private:
    SamplingConfig config_;
    std::mt19937_64 rng_;
};

} // namespace qwen
#endif
