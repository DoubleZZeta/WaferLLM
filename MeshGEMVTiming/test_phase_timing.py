import unittest
import numpy as np
import host_common as hc


class TimingTest(unittest.TestCase):
    def test_roles_and_skips(self):
        # P=8,G=4: groups of two, so no first-level bottom receive.
        self.assertEqual(hc.reduction_operations(5, 8, 4), [
            ('level1_root_receive_top', 8, 9),
            ('level2_root_receive_bottom', 10, 11),
            ('level2_root_receive_top', 12, 13)])
        # G=2: global root is in the last group, so no level-2 bottom receive.
        self.assertEqual(hc.reduction_operations(6, 8, 2), [
            ('level1_root_receive_bottom', 6, 7),
            ('level1_root_receive_top', 8, 9),
            ('level2_root_receive_top', 12, 13)])
        self.assertEqual(hc.reduction_operations(5, 16, 2),
                         [('level1_receive_add_send_up', 8, 9)])
        self.assertEqual(hc.reduction_operations(26, 32, 8)[-1],
                         ('level2_receive_add_send_up', 12, 13))

    def test_decoding_and_invalid_markers(self):
        P,G=8,2
        stamps=np.zeros((P,P,14),dtype=np.uint64)
        base=1<<36
        stamps[..., :6]=[base,base+10,base+110,0,base+410,base+420]
        stamps[6,:,3]=base+350
        for py in range(P):
            for label,start,end in hc.reduction_operations(py,P,G):
                stamps[py,:,start]=base+120+start*5
                stamps[py,:,end]=base+123+start*5
        raw=np.zeros((P,P,28),dtype=np.uint32)
        raw[...,::2]=(stamps & 0xffffffff).astype(np.uint32)
        raw[...,1::2]=(stamps>>32).astype(np.uint32)
        metrics,summaries,rows=hc.decode_phase_timing(raw.view(np.float32),
            np.zeros((P,P,2),dtype=np.float32),P,G,True)
        self.assertEqual(metrics['root_reduction']['mean_cycles'],240)
        self.assertEqual(metrics['root_broadcast']['mean_cycles'],60)
        self.assertEqual(metrics['dense_y_allreduce']['mean_cycles'],300)
        self.assertEqual(metrics['control_and_markers']['mean_cycles'],20)
        self.assertTrue(all(row['cycles']==3 for row in rows))
        self.assertEqual(len(rows),P*len(summaries))
        self.assertFalse(any(row['operation']=='level2_root_receive_bottom' for row in rows))
        with self.assertRaises(ValueError):
            hc.decode_phase_timing(raw[...,:12],None,P,G)

    def test_u48(self):
        self.assertEqual(hc.make_u48(np.array([65535]*3,dtype=np.uint16)),(1<<48)-1)


if __name__=='__main__':
    unittest.main()
