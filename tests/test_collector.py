"""Pure-logic collector helpers (no sockets/FIFO — I/O is exercised in E2E)."""
import unittest

from collector import Stats, block_attrs, build_expect_table, crunch


class TestBlockAttrs(unittest.TestCase):
    def test_pm_uses_key_basis_only(self):
        self.assertEqual(block_attrs("pm"), ("n_z", "m_z"))

    def test_ent_uses_all_sifted(self):
        # BBM92 keys both bases (Chapman: p=1/2 each, both generate key)
        self.assertEqual(block_attrs("ent"), ("sifted", "errors"))

    def test_attrs_are_real_stats_counters(self):
        # The block loop get/setattr's these names on a Stats instance.
        for protocol in ("pm", "ent"):
            for attr in block_attrs(protocol):
                self.assertIn(attr, Stats.FIELDS)


class TestCrunchPlainStream(unittest.TestCase):
    def test_plain_nibbles_count_as_signal(self):
        # Plain (non-decoy) stream: Alice nibble decoy bit is always 0.
        # 4 events: alice states 0,1,2,3 ; bob states 1,1,2,2 ; results 0.
        exp = build_expect_table([0, 32, 96, 64])
        aa = bytes([0x10, 0x32])   # nibbles 0,1,2,3 (low nibble first)
        ab = bytes([0x11, 0x22])   # nibbles 1,1,2,2
        rr = bytes([0, 0, 0, 0])
        s = crunch(aa, ab, rr, exp)
        self.assertEqual(s.clicks, 4)
        self.assertEqual(s.n_dec, 0)          # nothing labelled decoy
        self.assertEqual(s.n_sig, s.sifted)   # everything counts as signal
        self.assertGreater(s.sifted, 0)
        # sanity: sifted splits across both bases (alice 0,3 = Z; 1,2 = X)
        self.assertGreaterEqual(s.sifted, s.n_z)

    def test_x_basis_bits_are_sifted_but_not_key_basis(self):
        # Alice 1,2 (X) against Bob 0,3 sift, yet must not land in n_z:
        # this is exactly the PM/ENT difference the block accounting keys on.
        exp = build_expect_table([0, 32, 96, 64])
        aa = bytes([0x21])         # nibbles 1,2
        ab = bytes([0x30])         # nibbles 0,3
        rr = bytes([1, 1])
        s = crunch(aa, ab, rr, exp)
        self.assertEqual(s.sifted, 2)
        self.assertEqual(s.n_z, 0)
        self.assertEqual(s.m_z, 0)


if __name__ == "__main__":
    unittest.main()
