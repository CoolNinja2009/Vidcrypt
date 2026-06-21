#include "backend.h"
#include <string.h>

BackendMode backend_mode_from_string(const char *name) {
    (void)name;
    return BACKEND_CPU;
}

const char *backend_mode_name(BackendMode mode) {
    (void)mode;
    return "cpu";
}
