import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
import os

# Load NuScenes dataset
nusc = NuScenes(version='v1.0-mini', dataroot="v1.0-mini", verbose=True)

# Pick a scene (about 40 frames taken at 0.5 second intervals)
scene = nusc.scene[0]
first_sample = scene['first_sample_token']

# Collect all sample tokens in this scene
scene_samples = []
token = first_sample
while token:
    sample = nusc.get('sample', token)
    scene_samples.append(token)
    token = sample['next']

# Setup matplotlib figure with: LiDAR + Camera with annotated boxes
fig, (ax_lidar, ax_cam) = plt.subplots(
    1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1.5]}
)

# LiDAR axis
sc = ax_lidar.scatter([], [], c=[], cmap="Blues", s=0.5, vmin=-3, vmax=3)
ax_lidar.set_xlim(-50, 50)
ax_lidar.set_ylim(-50, 50)
ax_lidar.set_title("LiDAR Animation (Top-down, colored by height)")
ax_lidar.set_xlabel("x (m)")
ax_lidar.set_ylabel("y (m)")
ax_lidar.set_aspect("equal")

# Add colorbar for height
cbar = plt.colorbar(sc, ax=ax_lidar, fraction=0.046, pad=0.04)
cbar.set_label("Height (z, m)")

# Camera axis
ax_cam.axis("off")
ax_cam.set_title("Front Camera with Annotations")

def get_lidar_points(sample_token):
    """Load LiDAR point cloud for a given sample token in ego frame."""
    sample = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd_record['filename']))

    # Transform into ego car frame
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    return pc.points[0, :], pc.points[1, :], pc.points[2, :]  # x, y, z

def draw_front_camera(sample_token, ax):
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data']['CAM_FRONT']
    ax.clear()
    nusc.render_sample_data(cam_token, with_anns=True, ax=ax, verbose=False)
    ax.set_title("Front Camera with Annotations")

def init():
    sc.set_offsets(np.empty((0, 2)))
    sc.set_array(np.array([]))  # clear colors
    ax_cam.clear()
    ax_cam.axis("off")
    ax_cam.set_title("Front Camera with Annotations")
    return sc,

def update(frame_idx):
    current_token = scene_samples[frame_idx]

    # Update LiDAR
    x, y, z = get_lidar_points(current_token)
    coords = np.vstack((x, y)).T
    sc.set_offsets(coords)
    sc.set_array(z)
    ax_lidar.set_title(f"LiDAR Frame {frame_idx+1}/{len(scene_samples)} (colored by z)")

    # Update Camera with annotations
    draw_front_camera(current_token, ax_cam)

    return sc,

# Animation (plays automatically)
ani = animation.FuncAnimation(
    fig, update, frames=len(scene_samples),
    init_func=init, blit=False, interval=500  # 2 Hertz
)

plt.show()
