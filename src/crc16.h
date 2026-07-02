#ifndef VIDCRYPT_CRC16_H
#define VIDCRYPT_CRC16_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CRC-16-CCITT (poly 0x1021, init 0xFFFF) */
uint16_t crc16(const uint8_t *data, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* VIDCRYPT_CRC16_H */
