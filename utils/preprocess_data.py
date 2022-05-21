# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2021-2022, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys, getopt, os 

###==================================================================
#===============================================================
if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:t:") 
    except getopt.GetoptError:
        print('======================================')
        print('python preprocess_data.py -t datatype') #
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('======================================')
            print('Example usage (merge_nd_inputs4pred): python preprocess_data.py -t 0') 
            print('Example usage (make_ndwi_4pred): python preprocess_data.py -t 1') 
            print('Example usage (make_mndwi_4pred): python preprocess_data.py -t 2') 
            print('Example usage (make_ndwi_dataset): python preprocess_data.py -t 3') 
            print('Example usage (make_mndwi_dataset): python preprocess_data.py -t 4') 
            print('Example usage (vggjson2mask): python preprocess_data.py -t 5') 
            print('======================================')
            sys.exit()
        elif opt in ("-t"):
            data_type = arg
            data_type = int(data_type)

    if data_type==0:
        from doodleverse_utils import merge_nd_inputs4pred
    elif data_type==1:
        from doodleverse_utils import make_ndwi_4pred
    elif data_type==2:
        from doodleverse_utils import make_mndwi_4pred                
    elif data_type==3:
        from doodleverse_utils import make_ndwi_dataset
    elif data_type==4:
        from doodleverse_utils import make_mndwi_dataset
    elif data_type==5:
        from doodleverse_utils import vggjson2mask