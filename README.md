# graspable_objPose

### Dataset
- Download [GraspNet-1Billion](https://graspnet.net/datasets.html)
- Download [the the ground truth data for Point-wise Graspable Segmentation](https://feedu-my.sharepoint.com/:f:/g/personal/cuonghd12_fe_edu_vn/Eri6JqWRj5FOlF4IuxtAGhAB6wPg-3CszwhmQu0s9ogWjQ?e=BvWzk7)

## Requirements
- Python 3
- PyTorch 1.8
- Open3d 0.8
- TensorBoard 2.3
- NumPy
- SciPy
- Pillow
- tqdm

## Installation
Install graspnetAPI for using [GraspNet-1Billion dataset](https://graspnet.net/datasets.html).
```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI 
pip install .
```

### Visualize the ground truth data
```bash
python vis_graspability.py
```

### Training and Testing
Training and testing code will be available upon paper acceptance.
