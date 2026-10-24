"""Host-only tests for non-additive pipeline event decoding."""
import unittest
import numpy as np
import host_common as hc


def pack(times, base=0):
    stamps = (np.asarray(times, dtype=np.uint64) + np.uint64(base)) & np.uint64((1<<48)-1)
    raw = np.zeros(stamps.shape[:-1] + (hc.PROFILE_WORDS,), dtype=np.uint32)
    raw[..., 0::2] = stamps & np.uint64(0xffffffff)
    raw[..., 1::2] = stamps >> np.uint64(32)
    return raw.view(np.float32)


class PipelineProfileTests(unittest.TestCase):
    def test_overlapping_events_and_timestamp_wrap(self):
        # Deliberately non-monotonic event ID order: Y0 ready follows A1 end.
        times = [[[0,1,10,11,40,12,30,41,70,42,60,71,80,81,90,91]]]
        metrics, decoded = hc.decode_pipeline_timing(pack(times, (1<<48)-20))
        np.testing.assert_array_equal(decoded, times)
        self.assertEqual(metrics['Y0_A1_interval_overlap']['mean_cycles'], 18)
        self.assertEqual(metrics['Y1_B0_interval_overlap']['mean_cycles'], 18)
        self.assertEqual(metrics['profile_iteration']['mean_cycles'], 91)

    def test_serial_and_multiple_pe_durations(self):
        one = [0,1,10,31,40,11,30,61,70,41,60,71,80,81,90,91]
        times = np.array([[one, np.array(one)*2]])
        metrics, _ = hc.decode_pipeline_timing(pack(times))
        self.assertEqual(metrics['Y0_A1_interval_overlap']['mean_cycles'], 0)
        self.assertEqual(metrics['Y1_B0_interval_overlap']['mean_cycles'], 0)
        self.assertEqual(metrics['profile_iteration']['mean_cycles'], 136.5)
        self.assertEqual(metrics['profile_iteration']['max_pe_cycles'], 182)

    def test_bad_dependency_is_rejected(self):
        times = [[[0,1,10,11,80,12,30,41,70,42,60,71,90,91,100,101]]]
        with self.assertRaisesRegex(ValueError, 'bad dependency'):
            hc.decode_pipeline_timing(pack(times))

    def test_wrong_buffer_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.decode_pipeline_timing(np.zeros((4,4,14), dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
