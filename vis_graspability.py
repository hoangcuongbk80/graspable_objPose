import open3d as o3d
import scipy.io as scio
from PIL import Image
import os
import numpy as np
import sys
from utils.data_utils import get_workspace_mask, CameraInfo, create_point_cloud_from_depth_image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
data_path = '/media/cuong/HD-PZFU3/datasets/graspnet'
scene_id = 'scene_0030'
ann_id = '0000'
camera_type = 'kinect'
color = np.array(Image.open(os.path.join(data_path, 'scenes', scene_id, camera_type, 'rgb', ann_id + '.png')), dtype=np.float32) / 255.0
depth = np.array(Image.open(os.path.join(data_path, 'scenes', scene_id, camera_type, 'depth', ann_id + '.png')))
seg = np.array(Image.open(os.path.join(data_path, 'scenes', scene_id, camera_type, 'label', ann_id + '.png')))
meta = scio.loadmat(os.path.join(data_path, 'scenes', scene_id, camera_type, 'meta', ann_id + '.mat'))
intrinsic = meta['intrinsic_matrix']
factor_depth = meta['factor_depth']
camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
point_cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
depth_mask = (depth > 0)
camera_poses = np.load(os.path.join(data_path, 'scenes', scene_id, camera_type, 'camera_poses.npy'))
align_mat = np.load(os.path.join(data_path, 'scenes', scene_id, camera_type, 'cam0_wrt_table.npy'))
trans = np.dot(align_mat, camera_poses[int(ann_id)])
workspace_mask = get_workspace_mask(point_cloud, seg, trans=trans, organized=True, outlier=0.02)
mask = (depth_mask & workspace_mask)
point_cloud = point_cloud[mask]
color = color[mask]
seg = seg[mask]

graspability_full = np.load(os.path.join(data_path, 'graspability', scene_id, camera_type, ann_id + '.npy')).squeeze()
graspability_full[seg == 0] = 0.
print('graspability full scene: ', graspability_full.shape, (graspability_full > 0.1).sum())
color[graspability_full > 0.01] = [0., 1., 0.]


cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(point_cloud.astype(np.float32))
cloud.colors = o3d.utility.Vector3dVector(color.astype(np.float32))

vis1 = o3d.visualization.Visualizer()
vis1.create_window()
vis1.add_geometry(cloud)
opt1 = vis1.get_render_option()
opt1.point_size = 1
opt1.background_color = np.asarray([0, 0, 0])
vis1.run()
vis1.destroy_window()