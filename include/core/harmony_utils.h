#pragma once

#include <string>

namespace xllm::harmony {

std::string current_date();
std::string build_system_message();
std::string strip_control_tokens(std::string text);

}  // namespace xllm::harmony
