#include "sampler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace qwen {

// --- Configuration Validation ---
void SamplingConfig::validate() const {
    if (!std::isfinite(temperature) || temperature < 0.0f) {
        throw std::runtime_error("temperature must be finite and non-negative");
    }
    if (top_k < 0) {
        throw std::runtime_error("top-k must be non-negative");
    }
    if (!std::isfinite(top_p) || top_p <= 0.0f || top_p > 1.0f) {
        throw std::runtime_error("top-p must be in (0, 1]");
    }
    if (!std::isfinite(repetition_penalty) || repetition_penalty < 1.0f) {
        throw std::runtime_error("repeat penalty must be at least 1");
    }
    if (repetition_window < 0) {
        throw std::runtime_error("repeat-last-n must be non-negative");
    }
}

// --- Lifecycle ---
Sampler::Sampler(const SamplingConfig& config) : config_(config), rng_(config.seed) {
    config_.validate();
}

// --- Sampling Logic ---
int Sampler::sample(const float* logits, int vocab_size, const std::vector<int>& history) {
    if (!logits || vocab_size <= 0) {
        throw std::runtime_error("sampler received invalid logits");
    }

    std::vector<float> scores(logits, logits + vocab_size);

    // 1. Apply Repetition Penalty
    if (config_.repetition_penalty != 1.0f && config_.repetition_window > 0) {
        const size_t begin = history.size() > static_cast<size_t>(config_.repetition_window)
            ? history.size() - static_cast<size_t>(config_.repetition_window)
            : 0;

        std::unordered_set<int> seen;
        for (size_t i = begin; i < history.size(); ++i) {
            const int id = history[i];

            if (id >= 0 && id < vocab_size && seen.insert(id).second) {
                scores[id] = scores[id] < 0.0f
                             ? scores[id] * config_.repetition_penalty
                             : scores[id] / config_.repetition_penalty;
            }
        }
    }

    // Validate that logits are finite
    for (float score : scores) {
        if (!std::isfinite(score)) {
            throw std::runtime_error("sampler received non-finite logits");
        }
    }

    // 2. Fast-path: Greedy Decoding (Temperature = 0)
    if (config_.temperature == 0.0f) {
        return static_cast<int>(std::max_element(scores.begin(), scores.end()) - scores.begin());
    }

    struct Candidate {
        int id;
        float score;
        double probability;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(vocab_size);

    // 3. Apply Temperature
    for (int i = 0; i < vocab_size; ++i) {
        candidates.push_back({i, scores[i] / config_.temperature, 0.0});
    }

    // 4. Apply Top-K Filtering
    const int keep_k = config_.top_k == 0 ? vocab_size : std::min(config_.top_k, vocab_size);

    if (keep_k < vocab_size) {
        std::nth_element(
            candidates.begin(),
            candidates.begin() + keep_k,
            candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.score > b.score; }
        );
        candidates.resize(keep_k);
    }

    // Sort the surviving candidates
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const Candidate& a, const Candidate& b) { return a.score > b.score; }
    );

    // 5. Compute Softmax Probabilities
    const float maximum = candidates.front().score;
    double total = 0.0;

    for (auto& c : candidates) {
        c.probability = std::exp(static_cast<double>(c.score - maximum));
        total += c.probability;
    }
    for (auto& c : candidates) {
        c.probability /= total;
    }

    // 6. Apply Top-P (Nucleus) Filtering
    if (config_.top_p < 1.0f) {
        double cumulative = 0.0;
        size_t keep = 0;

        do {
            cumulative += candidates[keep++].probability;
        } while (keep < candidates.size() && cumulative < config_.top_p);

        candidates.resize(keep);
    }

    // 7. Sample from Final Distribution
    std::vector<double> weights;
    weights.reserve(candidates.size());

    for (const auto& c : candidates) {
        weights.push_back(c.probability);
    }

    std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());

    return candidates[distribution(rng_)].id;
}

} // namespace qwen
