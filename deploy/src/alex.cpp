#include "NvInfer.h"
#include "cuda_runtime.h"
#include "logging.h"
#include <map>
#include <chrono>
#include <fstream>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)