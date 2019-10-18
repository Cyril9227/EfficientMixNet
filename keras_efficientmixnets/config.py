# ==============================================================================
#
# Contains the BlockArgs class which encodes the arguments used by EfficientNet's blocks
#
# ==============================================================================

import re
from keras_efficientmixnets.functions_utils import get_activation_name, get_batchnorm_name


class BlockArgs(object):

    def __init__(self, input_filters=None,
                 output_filters=None,
                 kernel_size=1,
                 strides=None,
                 num_repeat=None,
                 se_ratio=None,
                 expand_ratio=None,
                 identity_skip=True,
                 activation='relu',
                 typeBN='bn',
                 n_mixture=None):

        
        if not(type(kernel_size) == list or type(kernel_size) == int):
            raise ValueError('kernel_size should be a list of integers or a single integer but is {} instead.'.format(type(kernel_size)))

        self.input_filters = input_filters
        self.output_filters = output_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.num_repeat = num_repeat
        self.se_ratio = se_ratio
        self.expand_ratio = expand_ratio
        self.identity_skip = identity_skip
        self.activation = activation
        self.typeBN = typeBN
        self.n_mixture = n_mixture
        

    def decode_block_string(self, block_string):
        """Instantiate a block through a string notation of arguments.
        
        Arguments:
            block_string {str} -- encoded arguments of the MBConvBlock 
        
        Raises:
            ValueError: if block_string is not a str
            ValueError: if strides are not a pair of integers (tuple or list)
        
        Returns:
            Instantiate BlockArgs object with encoded arguments provided by a string
        """

        if not isinstance(block_string, str):
            raise ValueError("block_string must be a string but got : {} instead".format(block_string))
            
        ops = block_string.split('_')
        options = {}
        for i, op in enumerate(ops):
            splits = re.split(r'(\d.*)', op)
            if i == 1:
                key, value = op[:1], eval(op[1:])
                options[key] = value
            else:
                if len(splits) >= 2:
                    key, value = splits[:2]
                    options[key] = value
                else:
                    key = splits[0][0]
                    value = splits[0][1:]
                    options[key] = value
                      

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')
            
        self.input_filters = int(options['i'])
        self.output_filters = int(options['o'])
        self.kernel_size = options['k']
        self.num_repeat = int(options['r'])
        self.identity_skip = ('noskip' not in block_string)
        self.se_ratio = float(options['se']) if 'se' in options else None
        self.expand_ratio = int(options['e'])
        self.strides = [int(options['s'][0]), int(options['s'][1])]
        self.activation = options['a']
        self.typeBN = options['b'] if 'b' in options else "bn"
        self.n_mixture = int(options['n']) if 'n' in options else None
        
        return self


    def encode_block_string(self, block):
        """Encode a block's arguments in a string
        
        Arguments:
            block {BlockArgs object} -- Block to encode
        
        Returns:
            [str] -- encoded arguments in string format
        """
   
        args = [
            'r%d' % block.num_repeat,
            'k{}'.format(block.kernel_size),
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters,
            'a{}'.format(block.activation),
            'b{}'.format(block.typeBN),
            'n{}'.format(block.n_mixture)
        ]

        if block.se_ratio > 0 and block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)

        if block.identity_skip is False:
            args.append('noskip')

        return '_'.join(args)

    @classmethod
    def from_block_string(cls, block_string):
        """
        Encoding Schema:
        "rX_kX_sXX_eX_iX_oX_aX_bX_nX{_se0.XX}{_noskip}"
         - X is replaced by a any number ranging from 0-9, by a list with the prefix k or by a string for prefix b and a
         - {} encapsulates optional arguments
            'r' : num_repeat,
            'k' : kernel_size(s),
            's' : strides,
            'e' : expand_ratio,
            'i' : input_filters,
            'o' : output_filters,
            'a' : activation,
            'b' : type of Batch Norm,
            'n' : n_mixture (only relevant for Attentive Normalization)

        To deserialize an encoded block string, use
        the class method :
        ```python
        BlockArgs.from_block_string(block_string)
        ```
        Arguments:
            block_string {str} -- Encoded args as string
        
        Returns:
            [BlockArgs] -- BlockArgs object initialized with the block string args.
        """
        
    
        block = cls()
        return block.decode_block_string(block_string)



# Default list of blocks for EfficientNets
def get_block_list(mixed=False, activation="swish", typeBN="bn", n_mixture=None):
    """Utility function to encode as strings the default block list of the NeuralNet for various paramaters combination 
    
    Arguments:
        mixed {bool} -- Whether to use mixed depthwise convolution instead of regular depthwise convolution
    
    Keyword Arguments:
        typeBN {str} -- type of batch normalization to use (bn : regular batchnorm, iebn : instance-enhancement bn, an : attentive bn) (default: {None})
        n_mixture {int} -- only relevant (default: {None})
        activation {str} -- type of activation function to use (default: {"relu"})
        
    Raises:
        ValueError: If using typeBN = "an" and not providing an integer for n_mixture
    
    Returns:
        [list of BlockArgs objects] -- Returns a list of block args objects to initialize MBConvBlock
    """
    activation = get_activation_name(activation)
    typeBN = get_batchnorm_name(typeBN)
    
    if typeBN == "an":
        if isinstance(n_mixture, int):
            typeBN = typeBN + "_n" + str(n_mixture)
        else:
            raise ValueError('When using typeBN="an", n_mixture must be an integer but got {} instead'.format(n_mixture))
          
    suffix = "_".join(["", "a" + activation, "b" + typeBN])
    
    if mixed:
        blocks_args = [
            'r1_k3_s11_e1_i32_o16_se0.25',
            'r2_k[3, 5, 7]_s22_e6_i16_o24_se0.25',
            'r2_k[5, 7, 9]_s22_e6_i24_o40_se0.25',
            'r3_k[3, 5, 7]_s22_e6_i40_o80_se0.25',
            'r3_k[3, 5, 7, 9]_s11_e6_i80_o112_se0.25',
            'r4_k[3, 5, 7, 9]_s22_e6_i112_o192_se0.25',
            'r1_k3_s11_e6_i192_o320_se0.25']
    else:
        blocks_args = [
            'r1_k3_s11_e1_i32_o16_se0.25',
            'r2_k3_s22_e6_i16_o24_se0.25',
            'r2_k5_s22_e6_i24_o40_se0.25',
            'r3_k3_s22_e6_i40_o80_se0.25',
            'r3_k5_s11_e6_i80_o112_se0.25',
            'r4_k5_s22_e6_i112_o192_se0.25',
            'r1_k3_s11_e6_i192_o320_se0.25']

    blocks_args = [i + suffix for i in blocks_args]  
    BLOCK_LIST = [BlockArgs.from_block_string(s) for s in blocks_args]

    return BLOCK_LIST



# default block list for efficientnets (original paper) :
# DEFAULT_BLOCK_LIST = [
#     BlockArgs(32, 16, kernel_size=3, strides=(1, 1), num_repeat=1, se_ratio=0.25, expand_ratio=1),
#     BlockArgs(16, 24, kernel_size=3, strides=(2, 2), num_repeat=2, se_ratio=0.25, expand_ratio=6),
#     BlockArgs(24, 40, kernel_size=5, strides=(2, 2), num_repeat=2, se_ratio=0.25, expand_ratio=6),
#     BlockArgs(40, 80, kernel_size=3, strides=(2, 2), num_repeat=3, se_ratio=0.25, expand_ratio=6),
#     BlockArgs(80, 112, kernel_size=5, strides=(1, 1), num_repeat=3, se_ratio=0.25, expand_ratio=6),
#     BlockArgs(112, 192, kernel_size=5, strides=(2, 2), num_repeat=4, se_ratio=0.25, expand_ratio=6),
#     BlockArgs(192, 320, kernel_size=3, strides=(1, 1), num_repeat=1, se_ratio=0.25, expand_ratio=6)]
# in encoded fashion :
# DEFAULT_BLOCK_LIST = [
#     'r1_k3_s11_e1_i32_o16_se0.25',
#     'r2_k3_s22_e6_i16_o24_se0.25',
#     'r2_k5_s22_e6_i24_o40_se0.25',
#     'r3_k3_s22_e6_i40_o80_se0.25',
#     'r3_k5_s11_e6_i80_o112_se0.25',
#     'r4_k5_s22_e6_i112_o192_se0.25',
#     'r1_k3_s11_e6_i192_o320_se0.25']

    
# default encoded block list for efficientnets using mixed depthwise convolutions
# DEFAULT_BLOCK_LIST = [
#     'r1_k3_s11_e1_i32_o16_se0.25',
#     'r2_k[3, 5, 7]_s22_e6_i16_o24_se0.25',
#     'r2_k[5, 7, 9]_s22_e6_i24_o40_se0.25',
#     'r3_k[3, 5, 7]_s22_e6_i40_o80_se0.25',
#     'r3_k[3, 5, 7, 9]_s11_e6_i80_o112_se0.25',
#     'r4_k[3, 5, 7, 9]_s22_e6_i112_o192_se0.25',
#     'r1_k3_s11_e6_i192_o320_se0.25']
