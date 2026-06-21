#ifndef VIDCRYPT_BACKEND_H
#define VIDCRYPT_BACKEND_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BACKEND_CPU = 1
} BackendMode;

BackendMode backend_mode_from_string(const char *name);
const char *backend_mode_name(BackendMode mode);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_BACKEND_H */
