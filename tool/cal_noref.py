import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from quality_no_refer import calculate_path_NRIQA

test_path = './enhanced_output/'


calculate_path_NRIQA(test_path)

