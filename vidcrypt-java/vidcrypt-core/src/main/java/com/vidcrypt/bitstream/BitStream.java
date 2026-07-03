package com.vidcrypt.bitstream;

/**
 * MSB-first bit packing/unpacking utilities.
 * Bits are stored in uint64_t words, MSB-first within each word.
 * Word 0 bits: [63=first bit, 62, ..., 0=64th bit]
 */
public final class BitStream {
    private BitStream() {}

    public static int bitsToBytes(byte[] bits, int nbits, byte[] bytes, int maxBytes) {
        int nbytes = nbits / 8;
        if (nbytes > maxBytes) nbytes = maxBytes;
        for (int i = 0; i < nbytes * 8; i++)
            if (bits[i] != 0) bytes[i / 8] |= (byte) (1 << (7 - (i % 8)));
        return nbytes;
    }

    public static void bytesToBits(byte[] bytes, int nbytes, byte[] bits, int maxBits) {
        int nbits = nbytes * 8;
        if (nbits > maxBits) nbits = maxBits;
        for (int i = 0; i < nbits; i++)
            bits[i] = (byte) ((bytes[i / 8] >> (7 - (i % 8))) & 1);
    }

    public static int bitsToWords(byte[] bits, int nbits, long[] words, int maxWords) {
        int nwords = (nbits + 63) / 64;
        if (nwords > maxWords) nwords = maxWords;
        for (int i = 0; i < nwords * 64 && i < nbits; i++)
            if (bits[i] != 0) words[i / 64] |= (1L << (63 - (i % 64)));
        return nwords;
    }

    public static void wordsToBits(long[] words, int nwords, byte[] bits, int maxBits) {
        int totalBits = nwords * 64;
        if (totalBits > maxBits) totalBits = maxBits;
        for (int i = 0; i < totalBits; i++)
            bits[i] = (byte) ((words[i / 64] >> (63 - (i % 64))) & 1);
    }

    public static long countBitsSet(long[] words, int nwords) {
        long count = 0;
        for (int i = 0; i < nwords; i++) count += Long.bitCount(words[i]);
        return count;
    }

    public static int getBit(long[] words, long index) {
        return (int) ((words[(int)(index / 64)] >> (63 - (int)(index % 64))) & 1);
    }

    public static void setBit(long[] words, long index, int value) {
        long mask = 1L << (63 - (int)(index % 64));
        int wi = (int)(index / 64);
        if (value != 0) words[wi] |= mask; else words[wi] &= ~mask;
    }

    public static void bitCopyRange(long[] src, long bitOffset, long nbits, long[] dst) {
        int dw = (int)((nbits + 63) / 64);
        for (int i = 0; i < dw; i++) dst[i] = 0;
        for (long i = 0; i < nbits; i++)
            if (getBit(src, bitOffset + i) != 0) setBit(dst, i, 1);
    }

    public static void bitCopyOffset(long[] src, long srcOffset,
                                      long[] dst, long dstOffset, long nbits) {
        for (long i = 0; i < nbits; i++) setBit(dst, dstOffset + i, getBit(src, srcOffset + i));
    }
}
