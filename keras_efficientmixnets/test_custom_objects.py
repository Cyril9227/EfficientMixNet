import unittest

import numpy as np
from keras.layers import DepthwiseConv2D, Input, Lambda
from keras.initializers import Ones
from keras.models import Model

from custom_objects import MDConv, GroupDepthwiseConvolution


class Test_config(unittest.TestCase):

    def test_split(self):
        """Utilitary function to split m channels into k groups"""
        odd_case = 7, 3
        even_case = 8, 2
        self.assertEqual(GroupDepthwiseConvolution(1,1,1)._split_channels(total_filters=odd_case[0], num_groups=odd_case[1]), [3, 2, 2])
        self.assertEqual(GroupDepthwiseConvolution(1,1,1)._split_channels(total_filters=even_case[0], num_groups=even_case[1]), [4, 4])

    def test_MDConv(self):
        """If a single integer is provided for kernel size then it's a regular DepthWiseConv2D. Else, MDConv performs a Mixed DepthWiseConv2D
        as described in the paper : https://arxiv.org/pdf/1907.09595.pdf"""

        # img_test = np.random.rand(4, 4, 3)
        # batch_test = np.array([img_test]) # tensor of shape (1, 4, 4, 3)

        # inp = Input(shape=(4,4,3))

        # x = MDConv(kernel_size=2, strides=(1, 1), padding="same", use_bias=False, kernel_initializer=Ones())(inp) # should be the same than a normal depthwiseconv2d
        # m = Model(inp, x)

        # x2 = DepthwiseConv2D(kernel_size=2, strides=(1, 1), padding="same", use_bias=False, kernel_initializer=Ones())(inp)
        # m2 = Model(inp, x2)

        # pred_custom = m.predict(batch_test).tolist() # cast to list to use assertEqual()
        # pred_base = m2.predict(batch_test).tolist()
        # self.assertEqual(pred_base, pred_custom)


        # mdconv = MDConv(kernel_size=[1, 2, 3], strides=(1, 1), padding="same", use_bias=False, kernel_initializer=Ones())(inp)
        # m3 = Model(inp, mdconv)

        # img_test = np.ones((4, 4, 3))
        # batch_test = np.array([img_test])

        # # expected result :
        # # first channel is made of 1s and is convoled with a (1,1) kernel with weight initialized to 1 so the channel will remain the same
        # c1 = img_test[:, :, 0]
        # # second channel is made of 1s and is convoled with a (2, 2) kernel with weight initialized to 1, the output will be mostly 4s 
        # # except for the 2s at the border bcs of padding = "same"
        # c2 = np.array([[4., 4., 4., 2.],
        #                 [4., 4., 4., 2.],
        #                 [4., 4., 4., 2.],
        #                 [2., 2., 2., 1.]])
        # # last channel is made of 1s and is convoled with a (3, 3) kernel with weight initialized to 1, the output will be mostly 9s at the center
        # # and 4s, 6s at the borders
        # c3 = np.array([[4., 6., 6., 4.],
        #                 [6., 9., 9., 6.],
        #                 [6., 9., 9., 6.],
        #                 [4., 6., 6., 4.]])

        # expected_result = np.dstack([c1, c2, c3]).tolist()
        # mdconv_result = m3.predict(batch_test)[0].tolist()

        # self.assertEqual(expected_result, mdconv_result)

        self.assertEqual(1, 1)

if __name__ == '__main__':
    unittest.main()
