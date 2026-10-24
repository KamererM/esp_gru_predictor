import unittest
from device_probe import int8,metadata_check

class ProbeTests(unittest.TestCase):
 def test_strict_int8(self):
  for v in [-128,0,127]:self.assertEqual(int8(v),v)
  for v in [True,False,1.0,"1",128,-129,None]:
   with self.assertRaises(ValueError):int8(v)
 def test_identity_and_arena(self):
  d={"run_id":"r","model_sha256":"m","app_elf_sha256":"e",
   "input_scale_float32_bits":"a","output_scale_float32_bits":"b",
   "input_zero_point":0,"output_zero_point":0,"arena_used_bytes":100,"arena_capacity_bytes":200}
  self.assertEqual(metadata_check(d,d),d)
  for k,v in [("run_id","bad"),("input_zero_point",1),("arena_used_bytes",201)]:
   with self.assertRaises(ValueError):metadata_check({**d,k:v},d)
 def test_resource_crosschecks(self):
  d={"run_id":"r","model_sha256":"m","app_elf_sha256":"e",
   "input_scale_float32_bits":"a","output_scale_float32_bits":"b",
   "input_zero_point":0,"output_zero_point":0,"arena_used_bytes":100,"arena_capacity_bytes":200,
   "resource_probe_version":"gru-resource-v1","internal_total_bytes":1000,
   "internal_free_now_bytes":600,"internal_min_free_since_boot_bytes":500,
   "internal_largest_free_block_bytes":300}
  metadata_check(d,d,True)
  with self.assertRaises(ValueError):
   metadata_check({**d,"internal_min_free_since_boot_bytes":700},d,True)
if __name__=="__main__":unittest.main()
