package com.vidcrypt.hash;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/** SHA-256 hash utilities using Java built-in MessageDigest. */
public final class Sha256 {
    public static final int DIGEST_LEN = 32;

    private Sha256() {}

    /** Compute SHA-256 of a file. */
    public static byte[] hashFile(Path path) throws IOException {
        MessageDigest md = getDigest();
        byte[] buf = new byte[65536];
        try (InputStream is = Files.newInputStream(path)) {
            int n;
            while ((n = is.read(buf)) > 0) {
                md.update(buf, 0, n);
            }
        }
        return md.digest();
    }

    /** Compute SHA-256 of byte data. */
    public static byte[] hashData(byte[] data, int offset, int len) {
        MessageDigest md = getDigest();
        md.update(data, offset, len);
        return md.digest();
    }

    /** Convenience: full array. */
    public static byte[] hashData(byte[] data) {
        return hashData(data, 0, data.length);
    }

    private static MessageDigest getDigest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("SHA-256 not available", e);
        }
    }
}
