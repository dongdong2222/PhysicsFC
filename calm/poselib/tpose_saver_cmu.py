import numpy as np
from addict import Dict
from collections import OrderedDict
from argparse import ArgumentParser
import sys
sys.path.append('/workspace/fbxbinding/build/Distrib/site-packages/fbx/')


def tpose_saver(path):
	'''
		this is only for cmu t-pose re-save
	'''

	file_name = path.split('.npy')[0]
	arr = np.load(path, allow_pickle=True)
	arr = Dict(arr.item())

	for k, v in arr.items():
		try:
			print(k, v.arr.shape)
		except:
			print(k, v)

	new_dict = Dict()
	new_dict.rotation.arr = arr.rotation.arr[0]
	new_dict.rotation.context = {'dtype': 'float32'}
	new_dict.root_translation.arr = arr.root_translation.arr[0]
	new_dict.root_translation.context = {'dtype': 'float32'}

	new_dict.skeleton_tree = \
	{
	'node_names': ['Hips',
	'LHipJoint',
	'LeftUpLeg',
	'LeftLeg',
	'LeftFoot',
	'LeftToeBase',
	'LeftToeBase_end',
	'RHipJoint',
	'RightUpLeg',
	'RightLeg',
	'RightFoot',
	'RightToeBase',
	'RightToeBase_end',
	'LowerBack',
	'Spine',
	'Spine1',
	'Neck',
	'Neck1',
	'Head',
	'Head_end',
	'LeftShoulder',
	'LeftArm',
	'LeftForeArm',
	'LeftHand',
	'LeftFingerBase',
	'LeftHandIndex1',
	'LeftHandIndex1_end',
	'LThumb',
	'LThumb_end',
	'RightShoulder',
	'RightArm',
	'RightForeArm',
	'RightHand',
	'RightFingerBase',
	'RightHandIndex1',
	'RightHandIndex1_end',
	'RThumb',
	'RThumb_end'],
	'parent_indices': {'arr': np.array([-1,  0,  1,  2,  3,  4,  5,  0,  7,  8,  9, 10, 11,  0, 13, 14, 15,
			16, 17, 18, 15, 20, 21, 22, 23, 24, 25, 23, 27, 15, 29, 30, 31, 32,
			33, 34, 32, 36]),
	'context': {'dtype': 'int64'}},
	'local_translation': {'arr': np.array([[ 1.7566456e-02,  8.4841949e-01,  1.7922592e+01],
			[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
			[ 0.0000000e+00,  2.4047999e+00,  0.0000000e+00],
			[ 2.2204460e-16,  7.1578002e+00,  0.0000000e+00],
			[ 0.0000000e+00,  7.4913998e+00, -5.5511151e-17],
			[-1.1102230e-16,  2.3678000e+00,  2.2204460e-16],
			[ 1.1102230e-16,  1.1897000e+00,  0.0000000e+00],
			[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
			[ 1.7763568e-15,  2.3727000e+00,  0.0000000e+00],
			[ 2.2204460e-16,  7.4338999e+00, -4.4408921e-16],
			[ 0.0000000e+00,  7.5091000e+00,  0.0000000e+00],
			[-2.2204460e-16,  2.4117000e+00,  0.0000000e+00],
			[ 1.1102230e-16,  1.2108001e+00,  0.0000000e+00],
			[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
			[-1.1102230e-16,  2.0450001e+00, -1.1102230e-16],
			[ 1.1102230e-16,  2.0501001e+00,  2.2204460e-16],
			[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
			[ 0.0000000e+00,  1.7555000e+00,  8.8817842e-16],
			[ 0.0000000e+00,  1.7688000e+00,  0.0000000e+00],
			[ 0.0000000e+00,  1.7719001e+00,  0.0000000e+00],
			[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
			[-3.5527137e-15,  3.5840001e+00, -2.2204460e-16],
			[ 0.0000000e+00,  4.9829998e+00, -2.2204460e-16],
			[-3.5527137e-15,  3.4835999e+00,  0.0000000e+00],
			[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
			[ 0.0000000e+00,  7.1530002e-01,  0.0000000e+00],
			[ 0.0000000e+00,  5.7669997e-01, -2.2204460e-16],
			[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
			[ 1.7763568e-15,  8.2800001e-01,  3.5527137e-15],
			[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
			[ 0.0000000e+00,  3.4482000e+00,  6.6613381e-16],
			[ 0.0000000e+00,  5.2419000e+00,  0.0000000e+00],
			[-3.5527137e-15,  3.4442000e+00, -2.2204460e-16],
			[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
			[ 0.0000000e+00,  6.2250000e-01,  2.2204460e-16],
			[ 0.0000000e+00,  5.0190002e-01,  2.2204460e-16],
			[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
			[ 1.7763568e-15,  7.2060001e-01,  3.5527137e-15]], dtype=np.float32),
	'context': {'dtype': 'float32'}},
	}

	new_dict.is_local = True
	new_dict['__name__'] = 'SkeletonState'
	new_dict = OrderedDict(new_dict)

	save_path = file_name + '_test' + '.npy'
	np.save(save_path, new_dict)
	print(save_path, 'saved')

if __name__ == '__main__':
	parser = ArgumentParser(description='you need input file path to re-save t pose npy file')
	parser.add_argument('file_path', type=str)
	args = parser.parse_args()
	
	path = args.file_path
	tpose_saver(path)