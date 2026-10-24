import unittest
import numpy as np
import host_common as hc


class RectangularHostTests(unittest.TestCase):
    def test_tile_assignment_and_factorized_product(self):
        for px, py in [(4, 8), (8, 4), (4, 4)]:
            with self.subTest(px=px, py=py):
                m, r, n = 64, 16, 128
                mt, rt, nt = m // py, r // px, n // py
                rng = np.random.default_rng(9)
                x = rng.normal(size=(1, m))
                a = rng.normal(size=(m, r))
                b = rng.normal(size=(r, n))
                xt = hc.tile_vector(x, px, py).reshape(py, px, mt)
                at = hc.tile_A_for_y_reduce(a, px, py, mt, rt).reshape(py, px, mt, rt)
                bt = hc.tile_B_for_x_reduce(b, px, py, rt, nt).reshape(py, px, rt, nt)
                for y in range(py):
                    for c in range(px):
                        np.testing.assert_array_equal(at[y, c], a[y*mt:(y+1)*mt, c*rt:(c+1)*rt])
                        np.testing.assert_array_equal(bt[y, c], b[c*rt:(c+1)*rt, y*nt:(y+1)*nt])
                z = np.einsum('yxi,yxij->yxj', xt, at).sum(axis=0)
                result = np.einsum('xr,yxrn->yxn', z, bt).sum(axis=1).reshape(1, n)
                np.testing.assert_allclose(result, x @ a @ b, rtol=1e-10, atol=1e-10)

    def test_rectangular_timestamps_with_high_words(self):
        for px, py in [(4, 8), (8, 4)]:
            times = np.zeros((py, px, 3), dtype=np.uint32)
            refs = np.zeros((py, px, 2), dtype=np.uint32)
            profile = np.zeros((py, px, 14), dtype=np.uint32)
            for y in range(py):
                for x in range(px):
                    ref = (1 << 36) + 1000 + x + y
                    start, end = ref + 100, ref + 300
                    times[y, x] = [start & 0xffffffff, ((start >> 32) & 0xffff) | ((end & 0xffff) << 16), end >> 16]
                    refs[y, x] = [ref & 0xffffffff, ref >> 32]
                    for k in range(7):
                        stamp = start + k * 10
                        profile[y, x, 2*k:2*k+2] = [stamp & 0xffffffff, stamp >> 32]
            mean, maximum = hc.decode_timing(times.view(np.float32), refs.view(np.float32), px, py, 2)
            self.assertEqual(mean, 100)
            self.assertEqual(maximum, (200 + px + py - 2) / 2)
            metrics = hc.decode_phase_timing(profile.view(np.float32), refs.view(np.float32), px, py)
            self.assertEqual(metrics['profile_iteration']['mean_cycles'], 60)
            self.assertEqual(metrics['stage1_local_compute']['mean_cycles'], 10)


if __name__ == '__main__':
    unittest.main()
