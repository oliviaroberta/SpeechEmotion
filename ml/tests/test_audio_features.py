import unittest,numpy as np
from audio_features import *
class Tests(unittest.TestCase):
 def setUp(self):self.c=load_feature_config();self.y=np.sin(np.linspace(0,1000,56000,dtype=np.float32))
 def test_shapes(self):
  self.assertEqual(extract_log_mel_spectrogram(self.y,config=self.c).shape,(64,219));self.assertEqual(extract_mfcc_matrix(self.y,config=self.c).shape,(13,219));self.assertEqual(extract_svm_feature_vector(self.y,config=self.c).shape,(78,))
 def test_properties(self):
  z=extract_svm_feature_vector(self.y*1.1,config=self.c);self.assertEqual(z.dtype,np.float32);self.assertTrue(z.flags.c_contiguous);self.assertTrue(np.isfinite(z).all());self.assertEqual(z.tobytes(),extract_svm_feature_vector(self.y*1.1,config=self.c).tobytes())
 def test_rejections(self):
  for y,sr in ((self.y[:-1],16000),(self.y,8000),(np.full(56000,np.nan),16000),(np.array([]),16000)):
   with self.assertRaises(ValueError):extract_mfcc_matrix(y,sr,self.c)
if __name__=='__main__':unittest.main()
