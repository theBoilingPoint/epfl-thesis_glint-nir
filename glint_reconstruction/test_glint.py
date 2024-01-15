import torch
import os
from dataclasses import dataclass
import nvdiffrast.torch as dr

from dataset.dataset_nerf import DatasetNERF

from train import prepare_batch, validate_itr
from geometry.dlmesh import DLMesh
from render import mesh, util, light

@dataclass
class FLAGS:
	"""
	Values that matter
	"""
	geometry: str
	material: str
	envmap: str
	dataset_path: str  # path to the data root folder where you have transforms_train.json and transforms_{phase}.json files
	cam_pos_script: str  # name of the camera position script
	out_dir: str # path to the output folder

	"""
	Some values from the config file. Please make sure they match.
	"""
	ks_min = [0.0, 0.0, 0.0]
	ks_max = [1.0, 1.0, 1.0]
	random_textures = True
	texture_res = [2048, 2048]
	train_res = [800, 800]
	dmtet_grid = 128
	mesh_scale = 2.1
	laplace_scale = 3000
	display = [{'latlong': True}, {'bsdf': 'kd'}, {'bsdf': 'ks'}, {'bsdf': 'normal'}]
	background = 'white'

	"""
	Values that don't matter.
	"""
	spp = 1
	layers = 1
	display_res = [800, 800]
	display_interval = 0
	min_roughness = 0.08
	custom_mip = False
	loss = 'logl1'
	validate = False
	isosurface = 'dmtet'
	mtl_override = None
	env_scale = 1.0
	camera_space_light = False
	lock_light = False
	lock_pos = False
	sdf_regularizer = 0.2
	laplace = 'relative'
	pre_load = True
	kd_min = [0.0, 0.0, 0.0, 0.0]
	kd_max = [1.0, 1.0, 1.0, 1.0]
	nrm_min= [-1.0, -1.0, 0.0]
	nrm_max = [1.0, 1.0, 1.0]
	cam_near_far = [0.1, 1000.0]
	learn_light = False
	local_rank = 0
	multi_gpu = False

	def __init__(self, geometry, material, envmap, dataset_path, cam_pos_script, out_dir):
		self.geometry = geometry
		self.material = material
		self.envmap = envmap
		self.dataset_path = dataset_path
		self.cam_pos_script = cam_pos_script
		self.out_dir = out_dir

def load_dataset(FLAGS):
	dataset = DatasetNERF(
		os.path.join(FLAGS.dataset_path, FLAGS.cam_pos_script), FLAGS
	)
	dataloader = torch.utils.data.DataLoader(
		dataset, batch_size=1, collate_fn=dataset.collate
	)

	return dataloader

def load_models(FLAGS):
	geometry_file = FLAGS.geometry
	lgt_file = FLAGS.envmap

	if geometry_file.endswith('.pt'):
		geometry = torch.load(geometry_file)
	else:
		base_mesh = mesh.load_mesh(geometry_file)
		geometry = DLMesh(base_mesh, FLAGS)

	if lgt_file.endswith('.pt'):
		lgt = torch.load(lgt_file)
	else:
		lgt = light.load_env(lgt_file, scale=FLAGS.env_scale)

	material = torch.load(FLAGS.material)
	print(
		f"Predicted Parameters -> _LogMicrofacetDensity: {material['glint_params'][0]}, _DensityRandomization: {material['glint_params'][1]}"
	)

	return geometry, material, lgt

if __name__ == "__main__":
	obj_lst = ["duck", "bob", "spot", 'sphere']
	obj = obj_lst[1]

	env_map_lst = ["clarens", "road", "golf"] if obj != "sphere" else ["e10", "e11", "e12", "e13", "e14"]
	env_map = env_map_lst[0]

	output_root_folder = f'out/final/{obj}'
	geometry_model_name = f'geometry_{env_map}.pt' if obj != "sphere" else f'geometry_{obj}.pt'
	material_model_name = f'material_{env_map}.pt'
	light_model_name = f'light_{env_map}.pt' if obj != "sphere" else f'light_{obj}.pt'

	flags = FLAGS(
		os.path.join(output_root_folder, geometry_model_name),
		os.path.join(output_root_folder, material_model_name),
		os.path.join(output_root_folder, light_model_name),
		f'data/nerf_synthetic/{obj}/{env_map}',
		'transforms_test.json', # Match this with the camera pose script in dataset_path
		os.path.join(output_root_folder, env_map)
	)

	os.makedirs(flags.out_dir, exist_ok=True)

	# load dataset
	dataloader = load_dataset(flags)

	# create rendering context
	glctx = dr.RasterizeGLContext()
	geometry, material, lgt = load_models(flags)

	for it, target in enumerate(dataloader):
		target = prepare_batch(target, flags.background)

		result_image, result_dict = validate_itr(
			glctx, target, geometry, material, lgt, flags
		)

		for k in result_dict.keys():
			# Only saving reference and rendered images.
			# Check validate_itr in train.py to see what other options are available.
			if k == "ref" or k == "opt":
				np_img = result_dict[k].detach().cpu().numpy()
				print(f'Saving {k} with index {it}...')
				util.save_image(flags.out_dir + "/" + ("glint_%03d_%s.png" % (it, k)), np_img)