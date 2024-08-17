import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import nvdiffrast.torch as dr

from dataset.dataset_nerf import DatasetNERF

from train import prepare_batch, validate_itr
from geometry.dlmesh import DLMesh
from render import mesh, util, light, texture

use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')

@dataclass
class FLAGS:
    """
    Values that matter
    """

    geometry: str
    material: str
    envmap: str
    pcg3d:str
    dataset_path: str  # path to the data root folder where you have transforms_train.json and transforms_{phase}.json files
    cam_pos_script: str  # name of the camera position script
    out_dir: str  # path to the output folder

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
    display = [{"latlong": True}, {"bsdf": "kd"}, {"bsdf": "ks"}, {"bsdf": "normal"}]
    background = "reference"

    """
	Values that don't matter.
	"""
    spp = 1
    layers = 1
    display_res = [800, 800]
    display_interval = 0
    min_roughness = 0.08
    custom_mip = False
    loss = "logl1"
    validate = False
    isosurface = "dmtet"
    mtl_override = None
    env_scale = 1.0
    camera_space_light = False
    lock_light = False
    lock_pos = False
    sdf_regularizer = 0.2
    laplace = "relative"
    pre_load = True
    kd_min = [0.0, 0.0, 0.0, 0.0]
    kd_max = [1.0, 1.0, 1.0, 1.0]
    nrm_min = [-1.0, -1.0, 0.0]
    nrm_max = [1.0, 1.0, 1.0]
    cam_near_far = [0.1, 1000.0]
    learn_light = False
    local_rank = 0
    multi_gpu = False

    def __init__(
        self, geometry, material, envmap, pcg3d, dataset_path, cam_pos_script, out_dir
    ):
        self.geometry = geometry
        self.material = material
        self.envmap = envmap
        self.pcg3d = pcg3d
        self.dataset_path = dataset_path
        self.cam_pos_script = cam_pos_script
        self.out_dir = out_dir


def load_dataset(FLAGS):
    dataset = DatasetNERF(os.path.join(FLAGS.dataset_path, FLAGS.cam_pos_script), FLAGS)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=dataset.collate
    )

    return dataloader


def load_models(FLAGS):
    geometry_file = FLAGS.geometry
    lgt_file = FLAGS.envmap
    mat_file = FLAGS.material
    pcg3d_file = FLAGS.pcg3d

    if geometry_file.endswith('.obj'):
        base_mesh = mesh.load_mesh(geometry_file)
        geometry = DLMesh(base_mesh, FLAGS)
    else:
        geometry = torch.load(geometry_file)
        
    if lgt_file.endswith('.hdr'):
        lgt = light.load_env(lgt_file, scale=FLAGS.env_scale)
    else:
        lgt = torch.load(lgt_file)
    
    if mat_file.endswith('.mtl'):
        # mat = material.load_mtl(mat_file)
        mat = {
            'name' : '_default_mat',
            'bsdf' : 'pbr',
            'kd'   : texture.Texture2D(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda')),
            'ks'   : texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda'))
        }
    else:
        mat_tmp = torch.load(FLAGS.material)
        pcg3dFloat = PCG3dFloat(3,3).to(device)
        pcg3dFloat.load_state_dict(torch.load(pcg3d_file))
        for param in pcg3dFloat.parameters():
            param.requires_grad = False
            
        mat = {
            'name' : '_default_mat',
            'bsdf' : 'pbr',
            'kd'   : texture.Texture2D(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda')),
            'ks'   : texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')),
            'glint_params': mat_tmp['glint_params'],
            'glint_pcg3d': pcg3dFloat,
            'glint_4d_noise': mat_tmp['glint_4d_noise']
        }
        # mat = torch.load(FLAGS.material)
        print(
            f"Predicted Parameters -> _LogMicrofacetDensity: {mat['glint_params'][0]}, _DensityRandomization: {mat['glint_params'][1]}"
        )

    return geometry, mat, lgt

class PCG3dFloat(nn.Module):
    def __init__(self, input_size, output_size):
        super(PCG3dFloat, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)

        return x


if __name__ == "__main__":
    # Edit this part ###############################
    obj = "bob"
    env_map = "clarens"

    # output_root_folder = (
    #     f"out/final/{obj}"  # match this with the output folder you have created
    # )
    output_root_folder = (
        f"out/test"  
    )
    # geometry_model_name = (
    #     f"geometry_{env_map}.pt"  # match this with the name of you saved geometry model
    # )
    geometry_model_name = 'bob.obj'
    material_model_name = (
        f"material_{env_map}.pt"  # match this with the name of you saved material model
    )
    # material_model_name = 'bob_tri.mtl'
    # light_model_name = (
    #     f"light_{env_map}.pt"  # match this with the name of you saved light model
    # )
    light_model_name = 'clarens.hdr'

    flags = FLAGS(
        os.path.join(output_root_folder, geometry_model_name),
        os.path.join(output_root_folder, material_model_name),
        os.path.join(output_root_folder, light_model_name),
        os.path.join(output_root_folder, 'model_state_dict.pth'),
        f"data/{obj}_{env_map}",  # match this with the dataset folder
        "transforms_test.json",  # match this with the camera pose script you want to use for the dataset you select above
        os.path.join(output_root_folder, env_map),
    )
    #################################################

    os.makedirs(flags.out_dir, exist_ok=True)

    # load dataset
    dataloader = load_dataset(flags)

    # create rendering context
    glctx = dr.RasterizeGLContext()
    geometry, mat, lgt = load_models(flags)

    for it, target in enumerate(dataloader):
        target = prepare_batch(target, flags.background)

        result_image, result_dict = validate_itr(
            glctx, target, geometry, mat, lgt, flags
        )

        for k in result_dict.keys():
            # Only saving reference and rendered images.
            # Check validate_itr in train.py to see what other options are available.
            if k == "ref" or k == "opt":
                np_img = result_dict[k].detach().cpu().numpy()
                print(f"Saving {k} with index {it}...")
                util.save_image(
                    flags.out_dir + "/" + ("glint_%03d_%s.png" % (it, k)), np_img
                )
