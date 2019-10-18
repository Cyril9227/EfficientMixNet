import unittest
from config import BlockArgs

class Test_config(unittest.TestCase):

    def test_init(self):
        """kernel_sizes should be initialised with a list or an integer"""
        with self.assertRaises(ValueError):
            BlockArgs(32, 16, kernel_size='[3,4,5]', strides=(1, 1), num_repeat=1, se_ratio=0.25, expand_ratio=1, identity_skip=False)
        
    def test_block_to_string(self):
        """Test that a block's arguments can be converted to a string"""
        block = BlockArgs(32, 16, kernel_size=3, strides=(1, 1), num_repeat=1, se_ratio=0.25, expand_ratio=1, identity_skip=False)
        block2 = BlockArgs(32, 16, kernel_size=[3,4,5], strides=(1, 1), num_repeat=1, se_ratio=0.25, expand_ratio=0.25, identity_skip=True)

        string = block.encode_block_string(block)
        string2 = block2.encode_block_string(block2)
        
        self.assertEqual(string, "r1_k3_s11_e1_i32_o16_se0.25_noskip")
        self.assertEqual(string2, "r1_k[3, 4, 5]_s11_e0.25_i32_o16_se0.25")

    def test_string_to_block_args(self):
        """Test that a block's arguments can be set according to a string"""
        block = BlockArgs(32, 16, kernel_size=3, strides=(1, 1), num_repeat=1, se_ratio=0.25, expand_ratio=1, identity_skip=False)
        self.assertEqual(block.kernel_size, 3)
        self.assertEqual(block.identity_skip, False)

        string = "r1_k[1,2,3]_s11_e1_i32_o16_se0.25"

        block = BlockArgs.from_block_string(string)
        self.assertEqual(block.kernel_size, [1,2,3])
        self.assertEqual(block.identity_skip, True)

    

if __name__ == '__main__':
    unittest.main()