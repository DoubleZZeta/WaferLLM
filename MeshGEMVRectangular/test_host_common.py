import unittest
import numpy as np
import host_common as hc


class DenseRectangularHostTests(unittest.TestCase):
    def test_tiling_reconstructs_gemv_for_all_orientations(self):
        for px, py in [(8,8),(4,16),(16,4)]:
            with self.subTest(px=px,py=py):
                np.random.seed(2025)
                m,n=64,128
                x,xt,w=hc.make_inputs(px,py,m,n)
                mt,nt=m//py,n//px
                wt=hc.tile_W(w,px,py,mt,nt).reshape(py,px,mt,nt)
                xt=xt.reshape(py,px,mt)
                for y in range(py):
                    for c in range(px):
                        np.testing.assert_array_equal(wt[y,c],w[y*mt:(y+1)*mt,c*nt:(c+1)*nt])
                        np.testing.assert_array_equal(xt[y,c],x[0,y*mt:(y+1)*mt])
                output=np.einsum('yxi,yxij->yxj',xt.astype(float),wt.astype(float)).sum(axis=0).reshape(1,n)
                np.testing.assert_allclose(output,x.astype(float)@w.astype(float))

    def test_high_timestamp_words_and_rectangular_phase_arrays(self):
        for px,py in [(4,16),(16,4)]:
            times=np.zeros((py,px,3),dtype=np.uint32)
            refs=np.zeros((py,px,2),dtype=np.uint32)
            profile=np.zeros((py,px,10),dtype=np.uint32)
            for y in range(py):
                for x in range(px):
                    ref=(1<<36)+1000+x+y
                    start,end=ref+100,ref+300
                    refs[y,x]=[ref&0xffffffff,ref>>32]
                    times[y,x]=[start&0xffffffff,((start>>32)&0xffff)|((end&0xffff)<<16),end>>16]
                    for k in range(5):
                        stamp=start+10*k
                        profile[y,x,2*k:2*k+2]=[stamp&0xffffffff,stamp>>32]
            mean,maximum=hc.decode_timing(times.view(np.float32),refs.view(np.float32),px,py,2)
            self.assertEqual(mean,100)
            self.assertEqual(maximum,(200+px+py-2)/2)
            phases=hc.decode_phase_timing(profile.view(np.float32),refs.view(np.float32),px,py)
            self.assertEqual(phases['profile_iteration']['mean_cycles'],40)
            self.assertEqual(phases['dense_local_compute']['mean_cycles'],10)


if __name__ == '__main__':
    unittest.main()
