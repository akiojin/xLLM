#pragma once

#include <cstdlib>
#include <cstdio>

#ifndef STCPP_DEBUG_LOG
#define STCPP_DEBUG_LOG(...) \
    do { \
        if (std::getenv("STCPP_DEBUG")) { \
            std::fprintf(stderr, __VA_ARGS__); \
            std::fflush(stderr); \
        } \
    } while (0)
#endif
