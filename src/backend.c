#include "backend.h"
#include <string.h>

BackendMode backend_mode_from_string(const char *name) {
    if (!name) return BACKEND_CPU;
    if (strcmp(name, "gpu")  == 0) return BACKEND_GPU;
    if (strcmp(name, "auto") == 0) return BACKEND_AUTO;
    return BACKEND_CPU;
}

const char *backend_mode_name(BackendMode mode) {
    switch (mode) {
        case BACKEND_GPU:  return "gpu";
        case BACKEND_AUTO: return "auto";
        default:           return "cpu";
    }
}
