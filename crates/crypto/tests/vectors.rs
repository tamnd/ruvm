// SPDX-License-Identifier: GPL-2.0-or-later

//! Known answer tests for the crypto primitives. The vectors are the ones in QEMU's
//! tests/unit/test-crypto-{cipher,hash,hmac,ivgen,pbkdf,afsplit}.c, which come from NIST, the
//! RFCs, the IEEE 1619 XTS test vectors and cryptsetup.

use ruvm_crypto::afsplit::{afsplit_decode, afsplit_encode};
use ruvm_crypto::cipher::{Cipher, cipher_get_block_len, cipher_get_iv_len, cipher_get_key_len};
use ruvm_crypto::hash::{Hash, hash_base64, hash_bytes, hash_bytesv, hash_digest, hash_digest_len};
use ruvm_crypto::hmac::Hmac;
use ruvm_crypto::ivgen::IvGen;
use ruvm_crypto::pbkdf::{pbkdf2, pbkdf2_count_iters_with_clock};
use ruvm_qapi::types::{
    QCryptoCipherAlgo as A, QCryptoCipherMode as M, QCryptoHashAlgo as H, QCryptoIVGenAlgo as I,
};

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

struct V {
    path: &'static str,
    alg: A,
    mode: M,
    key: &'static str,
    iv: Option<&'static str>,
    plaintext: Option<&'static str>,
    ciphertext: Option<&'static str>,
}

#[rustfmt::skip]
const CIPHER: &[V] = &[
    V { path: "/crypto/cipher/aes-ecb-128", alg: A::Aes128, mode: M::Ecb, key: "2b7e151628aed2a6abf7158809cf4f3c", iv: None, plaintext: Some("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e5130c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710"), ciphertext: Some("3ad77bb40d7a3660a89ecaf32466ef97f5d3d58503b9699de785895a96fdbaaf43b1cd7f598ece23881b00e3ed0306887b0c785e27e8ad3f8223207104725dd4") },
    V { path: "/crypto/cipher/aes-ecb-192", alg: A::Aes192, mode: M::Ecb, key: "8e73b0f7da0e6452c810f32b809079e562f8ead2522c6b7b", iv: None, plaintext: Some("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e5130c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710"), ciphertext: Some("bd334f1d6e45f25ff712a214571fa5cc974104846d0ad3ad7734ecb3ecee4eefef7afd2270e2e60adce0ba2face6444e9a4b41ba738d6c72fb16691603c18e0e") },
    V { path: "/crypto/cipher/aes-ecb-256", alg: A::Aes256, mode: M::Ecb, key: "603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4", iv: None, plaintext: Some("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e5130c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710"), ciphertext: Some("f3eed1bdb5d2a03c064b5a7e3db181f8591ccb10d410ed26dc5ba74a31362870b6ed21b99ca6f4f9f153e7b1beafed1d23304b7a39f9f3ff067d8d8f9e24ecc7") },
    V { path: "/crypto/cipher/aes-cbc-128", alg: A::Aes128, mode: M::Cbc, key: "2b7e151628aed2a6abf7158809cf4f3c", iv: Some("000102030405060708090a0b0c0d0e0f"), plaintext: Some("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e5130c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710"), ciphertext: Some("7649abac8119b246cee98e9b12e9197d5086cb9b507219ee95db113a917678b273bed6b8e3c1743b7116e69e222295163ff1caa1681fac09120eca307586e1a7") },
    V { path: "/crypto/cipher/aes-cbc-192", alg: A::Aes192, mode: M::Cbc, key: "8e73b0f7da0e6452c810f32b809079e562f8ead2522c6b7b", iv: Some("000102030405060708090a0b0c0d0e0f"), plaintext: Some("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e5130c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710"), ciphertext: Some("4f021db243bc633d7178183a9fa071e8b4d9ada9ad7dedf4e5e738763f69145a571b242012fb7ae07fa9baac3df102e008b0e27988598881d920a9e64f5615cd") },
    V { path: "/crypto/cipher/aes-cbc-256", alg: A::Aes256, mode: M::Cbc, key: "603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4", iv: Some("000102030405060708090a0b0c0d0e0f"), plaintext: Some("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e5130c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710"), ciphertext: Some("f58c4c04d6e5f1ba779eabfb5f7bfbd69cfc4e967edb808d679f777bc6702c7d39f23369a9d9bacfa530e26304231461b2eb05e2c39be9fcda6c19078c6a9d1b") },
    V { path: "/crypto/cipher/des-ecb-56-one-block", alg: A::Des, mode: M::Ecb, key: "80c4a2e691d5b3f7", iv: None, plaintext: Some("70617373776f7264"), ciphertext: Some("73fa80b66134e403") },
    V { path: "/crypto/cipher/des-cbc-56-one-block", alg: A::Des, mode: M::Cbc, key: "80c4a2e691d5b3f7", iv: Some("0000000000000000"), plaintext: Some("70617373776f7264"), ciphertext: Some("73fa80b66134e403") },
    V { path: "/crypto/cipher/des-ecb-56", alg: A::Des, mode: M::Ecb, key: "80c4a2e691d5b3f7", iv: None, plaintext: Some("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e5130c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710"), ciphertext: Some("8f346aaf64eaf24040720d80648c52e7aefc616be53ab1a3d301e69d91e01838ffd29f1bb5596ad94ea2d8e6196b7f0930d8ed0bf2773af36dd82a6280c20926") },
    V { path: "/crypto/cipher/3des-cbc", alg: A::V3des, mode: M::Cbc, key: "e9c0ff2e760b6424444d995a12d640c0eac284e81495dbe8", iv: Some("7d3388930f93b242"), plaintext: Some("6f54206f614d796e532063656572737454206f6f4d206e61207965537263746520736f54206f614d796e532063656572737454206f6f4d206e61207965537263746520736f54206f614d796e532063656572737454206f6f4d206e61207965537263746520736f54206f614d796e532063656572737454206f6f4d206e610a79"), ciphertext: Some("0e2db6973c5633f4671721c76e8ad54974b34905c51cd0ed12565c5396b6007d9048fcf58d2939cc8ad5351836234ed776d1da0c9467bb048bf2036ca8cfb6ea226447aa8f7513bf9fc2c3f0c956c57a71632e897b1e12cae25fafd8a4f8c97ad6f92131624445a6d6bc5ad32d5443cc9ddea570e942458a6bfab19113b0d919") },
    V { path: "/crypto/cipher/3des-ecb", alg: A::V3des, mode: M::Ecb, key: "0123456789abcdef5555555555555555fedcba9876543210", iv: None, plaintext: Some("736f6d6564617461"), ciphertext: Some("18d748e563620572") },
    V { path: "/crypto/cipher/3des-ctr", alg: A::V3des, mode: M::Ctr, key: "9cd6f39cb95a67005a67002dceeb2dceebb45172b451721f", iv: Some("ffffffffffffffff"), plaintext: Some("05ec77fb42d559208b128669f05bcf5639ad349f66ea7dc448d3ba0db118e34afe41285c278e11856cf75ec2553ca00b9265e970db4fd6b900b41fe649fd442f533a8d149863ca5dc1a833a70e9178ec77de42d5bc078b12e54cf05b225639806b9f66c950c4af36ba0d947fe34add4128b31a8e11f843f75e21553c876e9265cc57dba235b900eb72e649d0442fb6198d14ff46ca5d24a8339a6d9178c377dea108bc07ee71e54cd75b22b51c806bf245c9503baf369960947fc64adda40fb31aed74f8432a5e218813876ef158cc573ea2359c67eb72c549d0bb02b619e04bff46295d248f169a6df45fc3aa3da108937aee71d84cd7be01b51ce74ef2452c503b82159960cb52c6a930a40f9679ed74df432abd048813fa4df15823573e81689c67ce51c5ac37bb02957ce04bd24629b01b8f16f940f45f26aa3d846f937acd54d8a30abe01e873e74ed1452cb71e8215fc47cb5225a9309b629679c074dfa609bd04ef76fa4dd458238a1d8168f35ace5138ac379e61957cc74bd2a50cb01be275f9402b5f268910846ff659cd543fa30a9d64e873da4ed1b803b71ee148fc472e52258c179b62f55cc0ab32a609907bef76d94dd4bf068a1de44ff35a2d5138836a9e61c853c7ae31a50c977ee275dc402bb2058910fb42f65920543f86699d64cf56daad34b803ea7de148d347"), ciphertext: Some("07c20820721f49ef19cd6f3253052215a2852bdb85d2d8b9dd0d1b45cb6911d4eabeb2455d0caebea0c127ac659f537eafc21bb5b86d360c25c0f86d0b2901da1378dc89121243faf612ef8d87627883e2be41204c6d351bd10c30cfe2de2b03bf4573d4e55995d1b39b276297bdde7fa4d23980aa5023f074883da86a18793bc4966c8d2240926ed6ad2a1fde63c0e707f72df7b5f3f0cc017c2a9bc210caaafd2b3fc5f3f6fc9b45db53e45bf3c97b8e52ffc802b8ac9da10039da3d2d0e01097d8d5ebe53b9b08ee7e2966ab278eade238ba5fa5ce3dabf8e316a55d16ab2b5466fa5f0eeba1f9f98b0664fd03fa9df5f58c4f4ff755c403a097e6e1c97d4cce7e771cf0b150871fa0797cde6ca1d14280ccf99137af1ebfafa9207de1da1d33669fe514d9f2e83374f1f4830ed044da4ef3aca76f41c418f6337782f86a6ef417ed2af88ab675271c38ef8269372aad60ee70b46b13ab408a9a8a0cf200c52bc8b0556b2bc319b74b92929969a50dc45dc1aeb0c64d4d3057e5955c3f490c2abf89b8adacea1c3f4ad77dd44c8aca3f1c9d2195cb0caa234c1f76cfdac6532dc48c4f2006b77f17d76acc031632aa53a62c891b10365cb43d106dfc367bcdce0cd35ce4965a0527ba70d07a91bb0407772c2ea0e3a7846b991b6e73d5142fd51b0c62c6313785ceefccfc4700034") },
    V { path: "/crypto/cipher/cast5-128", alg: A::Cast5_128, mode: M::Ecb, key: "0123456712345678234567893456789A", iv: None, plaintext: Some("0123456789abcdef"), ciphertext: Some("238b4fe5847e44b2") },
    V { path: "/crypto/cipher/serpent-128", alg: A::Serpent128, mode: M::Ecb, key: "00000000000000000000000000000000", iv: None, plaintext: Some("d29d576fcea3a3a7ed9099f29273d78e"), ciphertext: Some("b2288b968ae8b08648d1ce9606fd992d") },
    V { path: "/crypto/cipher/serpent-192", alg: A::Serpent192, mode: M::Ecb, key: "000000000000000000000000000000000000000000000000", iv: None, plaintext: Some("d29d576fceaba3a7ed9899f2927bd78e"), ciphertext: Some("130e353e1037c22405e8faefb2c3c3e9") },
    V { path: "/crypto/cipher/serpent-256a", alg: A::Serpent256, mode: M::Ecb, key: "0000000000000000000000000000000000000000000000000000000000000000", iv: None, plaintext: Some("d095576fcea3e3a7ed98d9f29073d78e"), ciphertext: Some("b90ee5862de69168f2bdd5125b45472b") },
    V { path: "/crypto/cipher/serpent-256b", alg: A::Serpent256, mode: M::Ecb, key: "0000000000000000000000000000000000000000000000000000000000000000", iv: None, plaintext: Some("00000000010000000200000003000000"), ciphertext: Some("2061a42782bd52ec691ec383b03ba77c") },
    V { path: "/crypto/cipher/twofish-128", alg: A::Twofish128, mode: M::Ecb, key: "d491db16e7b1c39e86cb086b789f5419", iv: None, plaintext: Some("019f9809de1711858faac3a3ba20fbc3"), ciphertext: Some("6363977de839486297e661c6c9d668eb") },
    V { path: "/crypto/cipher/twofish-192", alg: A::Twofish192, mode: M::Ecb, key: "88b2b2706b105e36b446bb6d731a1e88efa71f788965bd44", iv: None, plaintext: Some("39da69d6ba4997d585b6dc073ca341b2"), ciphertext: Some("182b02d81497ea45f9daacdc29193a65") },
    V { path: "/crypto/cipher/twofish-256", alg: A::Twofish256, mode: M::Ecb, key: "d43bb7556ea32e46f2a282b7d45b4e0d57ff739d4dc92c1bd7fc01700cc8216f", iv: None, plaintext: Some("90afe91bb288544f2c32dc239b2635e6"), ciphertext: Some("6cb4561c40bf0a9705931cb6d408e7fa") },
    V { path: "/crypto/cipher/sm4", alg: A::Sm4, mode: M::Ecb, key: "0123456789abcdeffedcba9876543210", iv: None, plaintext: Some("0123456789abcdeffedcba9876543210"), ciphertext: Some("681edf34d206965e86b3e94f536e4246") },
    V { path: "/crypto/cipher/aes-xts-128-1", alg: A::Aes128, mode: M::Xts, key: "0000000000000000000000000000000000000000000000000000000000000000", iv: Some("00000000000000000000000000000000"), plaintext: Some("0000000000000000000000000000000000000000000000000000000000000000"), ciphertext: Some("917cf69ebd68b2ec9b9fe9a3eadda692cd43d2f59598ed858c02c2652fbf922e") },
    V { path: "/crypto/cipher/aes-xts-128-2", alg: A::Aes128, mode: M::Xts, key: "1111111111111111111111111111111122222222222222222222222222222222", iv: Some("33333333330000000000000000000000"), plaintext: Some("4444444444444444444444444444444444444444444444444444444444444444"), ciphertext: Some("c454185e6a16936e39334038acef838bfb186fff7480adc4289382ecd6d394f0") },
    V { path: "/crypto/cipher/aes-xts-128-3", alg: A::Aes128, mode: M::Xts, key: "fffefdfcfbfaf9f8f7f6f5f4f3f2f1f0bfbebdbcbbbab9b8b7b6b5b4b3b2b1b0", iv: Some("9a785634120000000000000000000000"), plaintext: Some("4444444444444444444444444444444444444444444444444444444444444444"), ciphertext: Some("b01f86f8edc1863706fa8a4253e34f28af319de38334870f4dd1f94cbe9832f1") },
    V { path: "/crypto/cipher/aes-xts-128-4", alg: A::Aes128, mode: M::Xts, key: "2718281828459045235360287471352631415926535897932384626433832795", iv: Some("00000000000000000000000000000000"), plaintext: Some("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9fa0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f8f9fafbfcfdfeff000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9fa0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f8f9fafbfcfdfeff"), ciphertext: Some("27a7479befa1d476489f308cd4cfa6e2a96e4bbe3208ff25287dd3819616e89cc78cf7f5e543445f8333d8fa7f56000005279fa5d8b5e4ad40e736ddb4d35412328063fd2aab53e5ea1e0a9f332500a5df9487d07a5c92cc512c8866c7e860ce93fdf166a24912b422976146ae20ce846bb7dc9ba94a767aaef20c0d61ad02655ea92dc4c4e41a8952c651d33174be51a10c421110e6d81588ede82103a252d8a750e8768defffed9122810aaeb99f9172af82b604dc4b8e51bcb08235a6f4341332e4ca60482a4ba1a03b3e65008fc5da76b70bf1690db4eae29c5f1badd03c5ccf2a55d705ddcd86d449511ceb7ec30bf12b1fa35b913f9f747a8afd1b130e94bff94effd01a91735ca1726acd0b197c4e5b03393697e126826fb6bbde8ecc1e08298516e2c9ed03ff3c1b7860f6de76d4cecd94c8119855ef5297ca67e9f3e7ff72b1e99785ca0a7e7720c5b36dc6d72cac9574c8cbbc2f801e23e56fd344b07f22154beba0f08ce8891e643ed995c94d9a69c9f1b5f499027a78572aeebd74d20cc39881c213ee770b1010e4bea718846977ae119f7a023ab58cca0ad752afe656bb3c17256a9f6e9bf19fdd5a38fc82bbe872c5539edb609ef4f79c203ebb140f2e583cb2ad15b4aa5b655016a8449277dbd477ef2c8d6c017db738b18deb4a427d1923ce3ff262735779a418f20a282df920147beabe421ee5319d0568") },
    V { path: "/crypto/cipher/cast5-xts-128", alg: A::Cast5_128, mode: M::Xts, key: "2718281828459045235360287471352631415926535897932384626433832795", iv: None, plaintext: None, ciphertext: None },
    V { path: "/crypto/cipher/aes-ctr-128", alg: A::Aes128, mode: M::Ctr, key: "2b7e151628aed2a6abf7158809cf4f3c", iv: Some("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff"), plaintext: Some("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e5130c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710"), ciphertext: Some("874d6191b620e3261bef6864990db6ce9806f66b7970fdff8617187bb9fffdff5ae4df3edbd5d35e5b4f09020db03eab1e031dda2fbe03d1792170a0f3009cee") },
    V { path: "/crypto/cipher/aes-ctr-192", alg: A::Aes192, mode: M::Ctr, key: "8e73b0f7da0e6452c810f32b809079e562f8ead2522c6b7b", iv: Some("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff"), plaintext: Some("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e5130c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710"), ciphertext: Some("1abc932417521ca24f2b0459fe7e6e0b090339ec0aa6faefd5ccc2c6f4ce8e941e36b26bd1ebc670d1bd1d665620abf74f78a7f6d29809585a97daec58c6b050") },
    V { path: "/crypto/cipher/aes-ctr-256", alg: A::Aes256, mode: M::Ctr, key: "603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4", iv: Some("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff"), plaintext: Some("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e5130c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710"), ciphertext: Some("601ec313775789a5b7a7f504bbf3d228f443e3ca4d62b59aca84e990cacaf5c52b0930daa23de94ce87017ba2d84988ddfc9c58db67aada613c2dd08457941a6") },
];

#[test]
fn cipher_vectors() {
    for v in CIPHER {
        let key = unhex(v.key);
        let r = Cipher::new(v.alg, v.mode, &key);
        let (Some(pt), Some(ct)) = (v.plaintext, v.ciphertext) else {
            assert!(r.is_err(), "{} should be refused", v.path);
            continue;
        };
        let mut c = r.unwrap_or_else(|e| panic!("{}: {}", v.path, e.message()));
        let keysize = cipher_get_key_len(v.alg);
        let niv = v.iv.map_or(0, |iv| iv.len() / 2);
        if v.mode == M::Xts {
            assert_eq!(keysize * 2, key.len(), "{}", v.path);
        } else {
            assert_eq!(keysize, key.len(), "{}", v.path);
        }
        assert_eq!(cipher_get_iv_len(v.alg, v.mode), niv, "{}", v.path);
        if niv != 0 {
            assert_eq!(cipher_get_block_len(v.alg), niv, "{}", v.path);
        }

        let mut buf = unhex(pt);
        if let Some(iv) = v.iv {
            c.setiv(&unhex(iv)).unwrap();
        }
        c.encrypt(&mut buf).unwrap();
        assert_eq!(hex(&buf), ct, "{} encrypt", v.path);

        let mut buf = unhex(ct);
        if let Some(iv) = v.iv {
            c.setiv(&unhex(iv)).unwrap();
        }
        c.decrypt(&mut buf).unwrap();
        assert_eq!(hex(&buf), pt, "{} decrypt", v.path);
    }
}

#[test]
fn cipher_null_iv() {
    // Encrypting without ever setting an IV uses an all zero IV rather than failing.
    let mut c = Cipher::new(A::Aes256, M::Cbc, &[0; 32]).unwrap();
    let mut buf = [0u8; 32];
    c.encrypt(&mut buf).unwrap();
}

#[test]
fn cipher_short_plaintext() {
    let mut c = Cipher::new(A::Aes256, M::Cbc, &[0; 32]).unwrap();
    let e = c.encrypt(&mut [0u8; 20]).unwrap_err();
    assert_eq!(e.message(), "Length 20 must be a multiple of block size 16");
    let e = c.encrypt(&mut [0u8; 40]).unwrap_err();
    assert_eq!(e.message(), "Length 40 must be a multiple of block size 16");
}

#[test]
fn cipher_errors() {
    let msg = |a, m, k: &[u8]| Cipher::new(a, m, k).unwrap_err().message().to_string();
    assert_eq!(msg(A::Aes128, M::Cbc, &[0; 15]), "Cipher key length 15 should be 16");
    assert_eq!(msg(A::Aes128, M::Xts, &[0; 31]), "XTS cipher key length should be a multiple of 2");
    assert_eq!(msg(A::Aes128, M::Xts, &[0; 16]), "Cipher key length 16 should be 32");
    assert_eq!(msg(A::Des, M::Xts, &[0; 16]), "XTS mode not compatible with DES/3DES");
    assert_eq!(msg(A::Cast5_128, M::Xts, &[0; 32]), "Unsupported cipher mode xts");
    let mut c = Cipher::new(A::Aes128, M::Cbc, &[0; 16]).unwrap();
    assert_eq!(c.setiv(&[0; 8]).unwrap_err().message(), "Expected IV size 16 not 8");
    let mut c = Cipher::new(A::Aes128, M::Ecb, &[0; 16]).unwrap();
    assert_eq!(c.setiv(&[0; 16]).unwrap_err().message(), "Setting IV is not supported");
}

const HASH_INPUT: &str = "Hiss hisss Hissss hiss Hiss hisss Hiss hiss";
const HASH_INPUT_PARTS: [&str; 3] = ["Hiss hisss ", "Hissss hiss ", "Hiss hisss Hiss hiss"];

#[rustfmt::skip]
const HASH: &[(H, &str, &str, usize)] = &[
    (H::Md5, "628d206371563035ab8ef62f492bdec9", "Yo0gY3FWMDWrjvYvSSveyQ==", 16),
    (H::Sha1, "b2e74f26758a3a421e509cee045244b78753cc02", "sudPJnWKOkIeUJzuBFJEt4dTzAI=", 20),
    (H::Sha224, "e2f7415aad33ef79f6516b0986d7175f9ca3389a85bf6cfed078737b", "4vdBWq0z73n2UWsJhtcXX5yjOJqFv2z+0Hhzew==", 28),
    (H::Sha256, "bc757abb0436586f392b437e5dd24096f7f224de6b74d4d86e2abc6121b160d0", "vHV6uwQ2WG85K0N+XdJAlvfyJN5rdNTYbiq8YSGxYNA=", 32),
    (H::Sha384, "887ce52efb4f46700376356583b7e2794f612bd024e4495087ddb946c448c69d56dbf7152a94a5e63a80f3ba9f0eed78", "iHzlLvtPRnADdjVlg7fieU9hK9Ak5ElQh925RsRIxp1W2/cVKpSl5jqA87qfDu14", 48),
    (H::Sha512, "3a90d79638235ec6c4c11bebd84d83c0549bc1e84edc4b6ec7086487641256cb63b54e4cb2d2032b393994aa263c0dbbe00a9f2fe9ef6037352232a1eec55ee7", "OpDXljgjXsbEwRvr2E2DwFSbwehO3Etuxwhkh2QSVstjtU5MstIDKzk5lKomPA274AqfL+nvYDc1IjKh7sVe5w==", 64),
    (H::Ripemd160, "f3d658fad3fdfb2b52c9369cf0d441249ddfa8a0", "89ZY+tP9+ytSyTac8NRBJJ3fqKA=", 20),
    (H::Sm3, "d4a97db105b477b84c4f20ec9c31a6c814e2705a0b83a5a89748d75f0ef456a1", "1Kl9sQW0d7hMTyDsnDGmyBTicFoLg6Wol0jXXw70VqE=", 32),
];

#[test]
fn hash_vectors() {
    for &(alg, hexd, b64, len) in HASH {
        assert_eq!(hash_digest_len(alg), len);
        assert_eq!(hex(&hash_bytes(alg, HASH_INPUT.as_bytes()).unwrap()), hexd, "{alg:?}");
        let parts: Vec<&[u8]> = HASH_INPUT_PARTS.iter().map(|s| s.as_bytes()).collect();
        // The iov variant hashes the concatenation, which is a different text.
        let joined = HASH_INPUT_PARTS.concat();
        assert_eq!(hash_bytesv(alg, &parts).unwrap(), hash_bytes(alg, joined.as_bytes()).unwrap());
        assert_eq!(hash_digest(alg, HASH_INPUT.as_bytes()).unwrap(), hexd);
        assert_eq!(hash_base64(alg, HASH_INPUT.as_bytes()).unwrap(), b64);
        let mut h = Hash::new(alg).unwrap();
        for p in HASH_INPUT_PARTS {
            h.update(p.as_bytes());
        }
        assert_eq!(h.finalize_bytes(), hash_bytes(alg, joined.as_bytes()).unwrap());
    }
}

const HMAC_KEY: &str = "monkey monkey monkey monkey";
const HMAC_PARTS: [&str; 3] =
    ["ABCDEFGHIJKLMNOPQRSTUVWXY", "Zabcdefghijklmnopqrstuvwx", "yz0123456789"];

#[rustfmt::skip]
const HMAC: &[(H, &str)] = &[
    (H::Md5, "ede9cb83679ba82d88fbeae865b3f8fc"),
    (H::Sha1, "c7b5a631e3aac975c4ededfcd346e469dbc5f2d1"),
    (H::Sha224, "5f768179dbb29ca722875d0f461a2e2f597d0210340a84df1a8e9c63"),
    (H::Sha256, "3798f363c57afa6edaffe39016ca7badefd1e670afb0e3987194307dec3197db"),
    (H::Sha384, "d218680a6032d33dccd9882d6a6a716464f26623be257a9b2919b185294f4a499e54b190bfd6bc5cedd2cd05c7e65e82"),
    (H::Sha512, "835a4f5b3750b4c1fccfa88da2f746a4900160c9f18964309bb736c13b59491b8e32d37b724cc5aebb0f554c6338a3b594c4ba26862b2dadb59b7ede1d08d53e"),
    (H::Ripemd160, "94964ed4c1155b62b668c241d67279e58a711676"),
    (H::Sm3, "760e3799332bc913819b930085360ddbc05529261313d5b15b75bab4fd7ae91e"),
];

#[test]
fn hmac_vectors() {
    let input = HMAC_PARTS.concat();
    for &(alg, want) in HMAC {
        let mut h = Hmac::new(alg, HMAC_KEY.as_bytes()).unwrap();
        assert_eq!(hex(&h.bytes(input.as_bytes()).unwrap()), want, "{alg:?}");
        // The key is kept, so a second message gets a fresh MAC.
        assert_eq!(h.digest(input.as_bytes()).unwrap(), want, "{alg:?}");
        let parts: Vec<&[u8]> = HMAC_PARTS.iter().map(|s| s.as_bytes()).collect();
        assert_eq!(h.digestv(&parts).unwrap(), want, "{alg:?}");
    }
}

#[rustfmt::skip]
/// Path, sector, generator, cipher, hash, key and expected IV.
type IvGenCase = (&'static str, u64, I, A, H, &'static [u8], &'static [u8]);

const IVGEN: &[IvGenCase] = &[
    (
        "plain/1",
        0x1,
        I::Plain,
        A::Aes128,
        H::Md5,
        b"",
        b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
    ),
    (
        "plain/1f2e3d4c",
        0x1f2e3d4c,
        I::Plain,
        A::Aes128,
        H::Md5,
        b"",
        b"\x4c\x3d\x2e\x1f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
    ),
    (
        "plain/1f2e3d4c5b6a7988",
        0x1f2e3d4c5b6a7988,
        I::Plain,
        A::Aes128,
        H::Md5,
        b"",
        b"\x88\x79\x6a\x5b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
    ),
    (
        "plain64/1",
        0x1,
        I::Plain64,
        A::Aes128,
        H::Md5,
        b"",
        b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
    ),
    (
        "plain64/1f2e3d4c",
        0x1f2e3d4c,
        I::Plain64,
        A::Aes128,
        H::Md5,
        b"",
        b"\x4c\x3d\x2e\x1f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
    ),
    (
        "plain64/1f2e3d4c5b6a7988",
        0x1f2e3d4c5b6a7988,
        I::Plain64,
        A::Aes128,
        H::Md5,
        b"",
        b"\x88\x79\x6a\x5b\x4c\x3d\x2e\x1f\x00\x00\x00\x00\x00\x00\x00\x00",
    ),
    (
        "essiv/1",
        0x1,
        I::Essiv,
        A::Aes128,
        H::Sha256,
        b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
        b"\xd4\x83\x71\xb2\xa1\x94\x53\x88\x1c\x7a\x2d\x06\x2d\x0b\x65\x46",
    ),
    (
        "essiv/1f2e3d4c",
        0x1f2e3d4c,
        I::Essiv,
        A::Aes128,
        H::Sha256,
        b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
        b"\x5d\x36\x09\x5d\xc6\x9e\x5e\xe9\xe3\x02\x8d\xd8\x7a\x3d\xe7\x8f",
    ),
    (
        "essiv/1f2e3d4c5b6a7988",
        0x1f2e3d4c5b6a7988,
        I::Essiv,
        A::Aes128,
        H::Sha256,
        b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
        b"\x58\xbb\x81\x94\x51\x83\x23\x23\x7a\x08\x93\xa9\xdc\xd2\xd9\xab",
    ),
];

#[test]
fn ivgen_vectors() {
    for &(path, sector, ivalg, cipher, hash, key, want) in IVGEN {
        let g = IvGen::new(ivalg, cipher, hash, key).unwrap();
        let mut iv = vec![0u8; want.len()];
        g.calculate(sector, &mut iv).unwrap();
        assert_eq!(iv, want, "{path}");
    }
}

struct P {
    path: &'static str,
    hash: H,
    iterations: u64,
    key: &'static [u8],
    salt: &'static [u8],
    out: &'static [u8],
    slow: bool,
}

#[rustfmt::skip]
const PBKDF: &[P] = &[
    P { path: "/crypto/pbkdf/rfc3962/sha1/iter1", hash: H::Sha1, iterations: 1, key: b"password", salt: b"ATHENA.MIT.EDUraeburn", out: b"\xcd\xed\xb5(\x1b\xb2\xf8\x01VZ\x11\x22\xb2V5\x15\x0a\xd1\xf7\xa0K\xb9\xf3\xa33\xec\xc0\xe2\xe1\xf7\x087", slow: false },
    P { path: "/crypto/pbkdf/rfc3962/sha1/iter2", hash: H::Sha1, iterations: 2, key: b"password", salt: b"ATHENA.MIT.EDUraeburn", out: b"\x01\xdb\xee\x7fJ\x9e$>\x98\x8bb\xc7<\xda\x93]\xa0Sx\xb92D\xec\x8fH\xa9\x9ea\xady\x9d\x86", slow: false },
    P { path: "/crypto/pbkdf/rfc3962/sha1/iter1200a", hash: H::Sha1, iterations: 1200, key: b"password", salt: b"ATHENA.MIT.EDUraeburn", out: b"\x5c\x08\xeba\xfd\xf7\x1eNN\xc3\xcfk\xa1\xf5Q+\xa7\xe5-\xdb\xc5\xe5\x14/p\x8a1\xe2\xe6+\x1e\x13", slow: false },
    P { path: "/crypto/pbkdf/rfc3962/sha1/iter5", hash: H::Sha1, iterations: 5, key: b"password", salt: b"\x124VxxV4\x12", out: b"\xd1\xda\xa7\x86\x15\xf2\x87\xe6\xa1\xc8\xb1 \xd7\x06*I?\x98\xd2\x03\xe6\xbeI\xa6\xad\xf4\xfaWKnd\xee", slow: false },
    P { path: "/crypto/pbkdf/rfc3962/sha1/iter1200b", hash: H::Sha1, iterations: 1200, key: b"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", salt: b"pass phrase equals block size", out: b"\x13\x9c0\xc0\x96k\xc3+\xa5_\xdb\xf2\x12S\x0a\xc9\xc5\xecY\xf1\xa4R\xf5\xcc\x9a\xd9@\xfe\xa0Y\x8e\xd1", slow: false },
    P { path: "/crypto/pbkdf/rfc3962/sha1/iter1200c", hash: H::Sha1, iterations: 1200, key: b"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", salt: b"pass phrase exceeds block size", out: b"\x9c\xca\xd6\xd4hw\x0c\xd5\x1b\x10\xe6\xa6\x87!\xbea\x1a\x8bM(&\x01\xdb;6\xbe\x92F\x91^\xc8*", slow: false },
    P { path: "/crypto/pbkdf/rfc3962/sha1/iter50", hash: H::Sha1, iterations: 50, key: b"\xf0\x9d\x84\x9e", salt: b"EXAMPLE.COMpianist", out: b"k\x9c\xf2mEEZC\xa5\xb8\xbb'j@;9\xe7\xfe7\xa0\xc4\x1e\x02\xc2\x81\xff0i\xe1\xe9OR", slow: false },
    P { path: "/crypto/pbkdf/rfc6070/sha1/iter1", hash: H::Sha1, iterations: 1, key: b"password", salt: b"salt", out: b"\x0c`\xc8\x0f\x96\x1f\x0eq\xf3\xa9\xb5$\xaf`\x12\x06/\xe07\xa6", slow: false },
    P { path: "/crypto/pbkdf/rfc6070/sha1/iter2", hash: H::Sha1, iterations: 2, key: b"password", salt: b"salt", out: b"\xeal\x01M\xc7-o\x8c\xcd\x1e\xd9*\xce\x1dA\xf0\xd8\xde\x89W", slow: false },
    P { path: "/crypto/pbkdf/rfc6070/sha1/iter4096", hash: H::Sha1, iterations: 4096, key: b"password", salt: b"salt", out: b"K\x00y\x01\xb7eH\x9a\xbe\xadI\xd9&\xf7!\xd0e\xa4)\xc1", slow: false },
    P { path: "/crypto/pbkdf/rfc6070/sha1/iter16777216", hash: H::Sha1, iterations: 16777216, key: b"password", salt: b"salt", out: b"\xee\xfe=a\xcdM\xa4\xe4\xe9\x94[=k\xa2\x15\x8c&4\xe9\x84", slow: true },
    P { path: "/crypto/pbkdf/rfc6070/sha1/iter4096a", hash: H::Sha1, iterations: 4096, key: b"passwordPASSWORDpassword", salt: b"saltSALTsaltSALTsaltSALTsaltSALTsalt", out: b"=.\xecO\xe4\x1c\x84\x9b\x80\xc8\xd86b\xc0\xe4J\x8b)\x1a\x96L\xf2\xf0p8", slow: false },
    P { path: "/crypto/pbkdf/rfc6070/sha1/iter4096b", hash: H::Sha1, iterations: 4096, key: b"pass\x00word", salt: b"sa\x00lt", out: b"V\xfaj\xa7UH\x09\x9d\xcc7\xd7\xf04%\xe0\xc3", slow: false },
    P { path: "/crypto/pbkdf/nonrfc/sha1/iter2", hash: H::Sha1, iterations: 2, key: b"", salt: b"salt", out: b"\x13:L\xe87\xb4\xd2R\x1e\xe2\xbf\x03\xe1\x1cq\xcayN\x07\x97", slow: false },
    P { path: "/crypto/pbkdf/nonrfc/sha256/iter1200", hash: H::Sha256, iterations: 1200, key: b"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", salt: b"pass phrase exceeds block size", out: b"\x224K\xc4\xb6\xe3&u\xa8\x09\x0f>\xa8\x0b\xe0\x1d_\x95\x12j,\xdd\xc3\xfa\xccJ^m\xca\x04\xecX", slow: false },
    P { path: "/crypto/pbkdf/nonrfc/sha512/iter1200", hash: H::Sha512, iterations: 1200, key: b"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", salt: b"pass phrase exceeds block size", out: b"\x0f\xb2\xed,\x0en\xfb}}\x8e\xddX\x01\xb4Yr\x99\x92\x160^\xa46\x8dv\x14\x80\xf3\xe3z\x22\xb9", slow: false },
    P { path: "/crypto/pbkdf/nonrfc/sha224/iter1200", hash: H::Sha224, iterations: 1200, key: b"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", salt: b"pass phrase exceeds block size", out: b"\x13;\x88\x0c\x0eR\xa2AI35\xa6\xc3\x83\xae#\xf6wC\x9e[0\x92>J:\xaa$i<\xed ", slow: false },
    P { path: "/crypto/pbkdf/nonrfc/sha384/iter1200", hash: H::Sha384, iterations: 1200, key: b"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", salt: b"pass phrase exceeds block size", out: b"\xfe\xe3\xe1\x84\xc9%>\x10G\xc8}S\xc6\xa5\xe3w)Av\xbdK\xe3\x9b\xac\x05l\x11\xdd\x17\xc5\x93\x80", slow: false },
    P { path: "/crypto/pbkdf/nonrfc/ripemd160/iter1200", hash: H::Ripemd160, iterations: 1200, key: b"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", salt: b"pass phrase exceeds block size", out: b"\xd6\xcb\xd8\xa7\xdb\x0c\xa2*#^G\xaf\xdb\xda\xa8\xef\xe4\x01\x0do\xb53\xc8\xbd\xce\xbf\x91\x14\x8b\x5cHA", slow: false },
    P { path: "/crypto/pbkdf/nonrfc/sm3/iter2", hash: H::Sm3, iterations: 2, key: b"password", salt: b"ATHENA.MIT.EDUraeburn", out: b"Hq\x1bX\xa3\xcb\xce\x06\xba\xadw\xa8\xb5\xb9\xd8\x07j\xe2\xb3[\x95\xce\xc8\xce\xe7\xb1\xcb\xeea\xdf\x04\xea", slow: false },
];

fn run_pbkdf(slow: bool) {
    for p in PBKDF.iter().filter(|p| p.slow == slow) {
        let mut out = vec![0u8; p.out.len()];
        pbkdf2(p.hash, p.key, p.salt, p.iterations, &mut out).unwrap();
        assert_eq!(hex(&out), hex(p.out), "{}", p.path);
    }
}

#[test]
fn pbkdf_vectors() {
    run_pbkdf(false);
}

/// The vectors QEMU only runs with `g_test_slow()`.
#[test]
#[ignore]
fn pbkdf_vectors_slow() {
    run_pbkdf(true);
}

#[test]
fn pbkdf_calibration() {
    // Each call returns the next reading. The first round of 1 << 15 iterations "takes" 50 ms,
    // under 100 ms, so the count grows tenfold; the second takes 640 ms, over 500 ms, which ends
    // the loop with 327680 iterations per 640 ms.
    let mut readings = [0u64, 50, 100, 740].into_iter();
    let mut clock = || Ok(readings.next().unwrap());
    let iters =
        pbkdf2_count_iters_with_clock(H::Sha256, &[0x5d; 32], &[0x7c; 32], 32, &mut clock).unwrap();
    assert_eq!(iters, 327680 * 1000 / 640);

    // Between 100 and 500 ms the count is rescaled to aim at one second.
    let mut readings = [0u64, 250, 1000, 1600].into_iter();
    let mut clock = || Ok(readings.next().unwrap());
    let iters =
        pbkdf2_count_iters_with_clock(H::Sha256, &[0x5d; 32], &[0x7c; 32], 32, &mut clock).unwrap();
    assert_eq!(iters, (32768 * 1000 / 250) * 1000 / 600);
}

struct F {
    path: &'static str,
    hash: H,
    stripes: u32,
    blocklen: usize,
    key: &'static [u8],
    splitkey: Option<&'static [u8]>,
}

#[rustfmt::skip]
const AFSPLIT: &[F] = &[
    F { path: "/crypto/afsplit/sha256/5", hash: H::Sha256, stripes: 5, blocklen: 32, key: b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf", splitkey: Some(b"\xfd\xd2s\xb1}\x99\x934p\xde\xfa\x07\xc5\xacX\xd20g/\x1a5C`}w\x02\xdbb<\xcb,3H\x08\xb6\xf1|\xa3 \xa0\xad-L\xf3\xcd\x18oS\xf9\xe8\xe7Y'<\xa9Ta\x87\xb3\xaf\xf6\xf7~d\x86\xaa\x89\x7f\x1f\x9f\xdb\x86\xf4\xa2\x16\xff\xa3O\x8c\xa1Y\xc4#4(\xc4wq\x83\xd4\xcd\x8e\x89\x1b\xc7\xc5\xaeM\xa9\xcd\xc9r\x85p\x13hR\x83\xfc\xb8\x11r\xba=\xc6J(\xfa\xe2\x86{'\xabX\xe1\xa4\xca\xf6\x9e\xbc\xfe\x0c\x92y\xb3\xec\x1c_y;\x0d\x1e\xaa\x1aw\x0fp\x19K\xc8\x80\xee'|nJ\x91\x96\x5c\xf4") },
    F { path: "/crypto/afsplit/sha256/5000", hash: H::Sha256, stripes: 5000, blocklen: 16, key: b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f", splitkey: None },
    F { path: "/crypto/afsplit/sha1/1000", hash: H::Sha1, stripes: 1000, blocklen: 32, key: b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf", splitkey: None },
    F { path: "/crypto/afsplit/sha256/big", hash: H::Sha256, stripes: 1000, blocklen: 64, key: b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f", splitkey: None },
];

#[test]
fn afsplit_vectors() {
    for f in AFSPLIT {
        let mut split = vec![0u8; f.blocklen * f.stripes as usize];
        let mut key = vec![0u8; f.blocklen];
        afsplit_encode(f.hash, f.blocklen, f.stripes, f.key, &mut split).unwrap();
        afsplit_decode(f.hash, f.blocklen, f.stripes, &split, &mut key).unwrap();
        assert_eq!(key, f.key, "{} round trip", f.path);
        if let Some(sk) = f.splitkey {
            key.fill(0);
            afsplit_decode(f.hash, f.blocklen, f.stripes, sk, &mut key).unwrap();
            assert_eq!(key, f.key, "{} decode", f.path);
        }
    }
}
