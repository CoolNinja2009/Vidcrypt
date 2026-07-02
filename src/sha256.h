#ifndef VIDCRYPT_SHA256_H
#define VIDCRYPT_SHA256_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SHA256_DIGEST_LEN 32

void sha256_file(const char *path, uint8_t hash[32]);
void sha256_data(const uint8_t *data, size_t len, uint8_t hash[32]);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_SHA256_H */
