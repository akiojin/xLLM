#include "core/harmony_utils.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <sstream>
#include <vector>

namespace xllm::harmony {

std::string current_date() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[11] = {0};
    if (std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm) == 0) {
        return "1970-01-01";
    }
    return std::string(buf);
}

std::string build_system_message() {
    std::ostringstream oss;
    oss << "<|start|>system<|message|>"
        << "You are ChatGPT, a large language model trained by OpenAI.\n"
        << "Knowledge cutoff: 2024-06\n"
        << "Current date: " << current_date() << "\n\n"
        << "Reasoning: low\n\n"
        << "# Valid channels: analysis, commentary, final. Channel must be included for every message.\n"
        << "<|end|>\n";
    return oss.str();
}

std::string strip_control_tokens(std::string text) {
    const std::vector<std::string> tokens = {
        "<|start|>", "<|end|>", "<|message|>", "<|channel|>",
        "<|return|>", "<|constrain|>",
        "<|im_start|>", "<|im_end|>", "<s>", "</s>", "<|endoftext|>", "<|eot_id|>"
    };
    for (const auto& token : tokens) {
        size_t pos = 0;
        while ((pos = text.find(token, pos)) != std::string::npos) {
            text.erase(pos, token.size());
        }
    }
    auto l = text.find_first_not_of(" \t\n\r");
    if (l == std::string::npos) return "";
    auto r = text.find_last_not_of(" \t\n\r");
    return text.substr(l, r - l + 1);
}

}  // namespace xllm::harmony
