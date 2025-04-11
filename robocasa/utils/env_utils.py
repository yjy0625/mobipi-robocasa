from robocasa.utils.dataset_registry import (
    get_ds_path,
    SINGLE_STAGE_TASK_DATASETS,
    MULTI_STAGE_TASK_DATASETS,
)
from robocasa.scripts.playback_dataset import get_env_metadata_from_dataset
from robosuite.controllers import load_composite_controller_config
from scipy.spatial.transform import Rotation as R
import os
import robosuite
import imageio
import numpy as np
from tqdm import tqdm
from termcolor import colored
from shapely.geometry import Polygon, LineString, Point


def create_env(
    env_name,
    # robosuite-related configs
    robots="PandaOmron",
    camera_names=[
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ],
    camera_widths=128,
    camera_heights=128,
    seed=None,
    # robocasa-related configs
    obj_instance_split=None,
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=None,
    layout_ids=None,
    style_ids=None,
    place_robot=True,
    place_robot_for_nav=False,
):
    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots if isinstance(robots, str) else robots[0],
    )

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_config,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        layout_ids=layout_ids,
        style_ids=style_ids,
        translucent_robot=False,
        place_robot=place_robot,
        place_robot_for_nav=place_robot_for_nav,
    )

    env = robosuite.make(**env_kwargs)
    return env


def run_random_rollouts(env, num_rollouts, num_steps, video_path=None):
    video_writer = None
    if video_path is not None:
        video_writer = imageio.get_writer(video_path, fps=20)

    info = {}
    num_success_rollouts = 0
    for rollout_i in tqdm(range(num_rollouts)):
        obs = env.reset()
        for step_i in range(num_steps):
            # sample and execute random action
            action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
            obs, _, _, _ = env.step(action)

            if video_writer is not None:
                video_img = env.sim.render(
                    height=512, width=512, camera_name="robot0_agentview_center"
                )[::-1]
                video_writer.append_data(video_img)

            if env._check_success():
                num_success_rollouts += 1
                break

    if video_writer is not None:
        video_writer.close()
        print(colored(f"Saved video of rollouts to {video_path}", color="yellow"))

    info["num_success_rollouts"] = num_success_rollouts

    return info


def get_robot_xy_corners(pose, robot_size=np.array([0.25, 0.35])):
    """
    Get the corners of the robot in the xy-plane based on its pose.

    Args:
        pose (list): [x, y, heading] of the robot.
        robot_size (tuple): (width, length) of the robot.

    Returns:
        np.ndarray: 4x2 array of corner coordinates.
    """
    x, y, heading = pose
    width, length = robot_size
    heading_vec = np.array([np.cos(pose[-1]), np.sin(pose[-1])])
    xy_center = pose[:2] - 0.21 * heading_vec

    # Local corners in the robot frame
    local_corners = np.array(
        [
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
            [length / 2, width / 2],
            [length / 2, -width / 2],
        ]
    )

    # Rotate and translate corners to the world frame
    rotation_matrix = np.array(
        [
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)],
        ]
    )
    world_corners = (rotation_matrix @ local_corners.T).T + xy_center

    return world_corners


def grid_based_pose_sampling(
    base_fixture_bounds_2d,
    floor_fixture_bounds_2d,
    default_robot_pos,
    rng,
    robot_size=np.array([0.25, 0.35]),
    grid_resolution=0.1,
    orientation_resolution=np.pi / 12,
    fov=np.pi * 75 / 180,
    max_dist=3.0,
):
    """
    Sample a valid robot pose using a grid-based approach with optional visualization.

    Args:
        base_fixture_bounds_2d (list of np.ndarray): List of rectangular occupied areas.
        floor_fixture_bounds_2d (np.ndarray): Floor area defined by 4 corner points.
        default_robot_pos (np.ndarray): Default robot position as a 2D array.
        rng (np.random.Generator): Random number generator.
        robot_size (tuple): Robot dimensions (width, length).
        grid_resolution (float): Resolution of the grid for x and y (meters).
        orientation_resolution (float): Resolution of orientation sampling (radians).
        fov (float): Field of view in radians (centered around the robot's forward direction).

    Returns:
        np.ndarray: A valid robot pose as [x, y, orientation].
    """
    # Convert obstacles and floor bounds into Shapely polygons
    occupied_polygons = [Polygon(bounds) for bounds in base_fixture_bounds_2d]
    floor_polygon = Polygon(floor_fixture_bounds_2d)

    # Precompute grid points for the floor
    min_x, min_y, max_x, max_y = floor_polygon.bounds
    x_coords = np.arange(min_x, max_x, grid_resolution)
    y_coords = np.arange(min_y, max_y, grid_resolution)
    orientations = np.arange(-np.pi, np.pi, orientation_resolution)

    # Initialize heatmap for feasible headings
    heatmap = np.zeros((len(x_coords), len(y_coords)))

    # Initialize a list to store valid poses
    valid_poses = []

    # Iterate through the grid and orientations
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            point = Point(x, y)
            if not floor_polygon.contains(point):
                continue

            feasible_orientations = 0
            for orientation in orientations:
                # Check if the pose is too far away
                if np.linalg.norm(np.array([x, y]) - default_robot_pos) > max_dist:
                    continue

                # Check if the pose is collision-free
                pose = [x, y, orientation]
                robot_polygon = Polygon(get_robot_xy_corners(pose, robot_size))
                if not floor_polygon.contains(robot_polygon):
                    continue
                if any(
                    polygon.intersects(robot_polygon) for polygon in occupied_polygons
                ):
                    continue

                # Check if the robot is collision-free if it directly
                # approaches the target (aka. default pose)
                nav_vec = default_robot_pos - pose[:2]
                nav_heading = np.arctan2(nav_vec[1], nav_vec[0])
                init_corners_during_nav = get_robot_xy_corners(
                    [x, y, nav_heading], robot_size
                )
                target_corners_during_nav = get_robot_xy_corners(
                    [*default_robot_pos, nav_heading], robot_size
                )
                valid = True
                for corner_start, corner_end in zip(
                    init_corners_during_nav, target_corners_during_nav
                ):
                    line_to_default = LineString([corner_start, corner_end])
                    if any(
                        polygon.intersects(line_to_default)
                        for polygon in occupied_polygons
                    ):
                        valid = False
                        break
                if not valid:
                    continue

                # Check if the default position is within FOV
                delta = default_robot_pos - np.array([x, y])
                angle_to_default = np.arctan2(delta[1], delta[0])
                min_angle = angle_to_default - fov / 2
                max_angle = angle_to_default + fov / 2
                if min_angle <= orientation <= max_angle:
                    valid_poses.append(pose)
                    feasible_orientations += 1

            # Update heatmap with the number of feasible orientations
            heatmap[i, j] = feasible_orientations

    # Randomly select one of the valid poses
    if not valid_poses:
        raise ValueError("No valid poses found with the given constraints.")
    selected_pose = np.array(rng.choice(valid_poses))

    return selected_pose


def get_base_pose(env, unwrapped=False):
    if not unwrapped:
        unwrapped_env = env.unwrapped.env
    else:
        unwrapped_env = env
    base_pos, base_mat = unwrapped_env.robots[
        0
    ].composite_controller.get_controller_base_pose("right")
    heading = R.from_matrix(base_mat).as_euler("xyz")[-1]
    return np.array([base_pos[0], base_pos[1], heading])


def set_robot_base_pose(env, xy_heading):
    """
    Sets the robot's base pose to the specified absolute position and heading,
    considering joint offsets in the parent frame.

    Args:
        env: The simulation environment.
        xy_heading: A list or array [x, y, heading] specifying the absolute position (x, y)
                    and heading angle (yaw) in world coordinates.
    """
    # Extract x, y, and heading (yaw) from input
    x, y, heading = xy_heading

    # Get the parent body's world position and orientation (robot0_base)
    parent_body_name = "robot0_base"
    parent_body_id = env.sim.model.body_name2id(parent_body_name)
    parent_pos = env.sim.data.body_xpos[parent_body_id][
        :2
    ]  # Parent position (x, y) in world frame
    parent_quat = env.sim.data.body_xquat[
        parent_body_id
    ]  # Parent orientation (quaternion) in world frame

    # Adjust for the joint offset
    forward_joint_offset = env.sim.model.jnt_pos[
        env.sim.model.joint_name2id("mobilebase0_joint_mobile_forward")
    ][:2]
    side_joint_offset = env.sim.model.jnt_pos[
        env.sim.model.joint_name2id("mobilebase0_joint_mobile_side")
    ][:2]
    joint_offset = np.array([side_joint_offset[0], forward_joint_offset[1]])

    # Compute the relative position in the world frame
    parent_rot = R.from_quat(
        parent_quat[[1, 2, 3, 0]]
    )  # Convert parent quaternion to rotation matrix
    joint_offset_rot_target = R.from_euler("z", heading).apply(
        np.array([joint_offset[0], joint_offset[1], 0])
    )[:2]
    joint_offset_rot_source = parent_rot.apply(
        np.array([joint_offset[0], joint_offset[1], 0])
    )[:2]
    abs_pos = np.array([x, y])  # Absolute position in world coordinates
    rel_pos_world = (abs_pos + joint_offset_rot_target) - (
        parent_pos + joint_offset_rot_source
    )  # Relative position in the world frame

    # Rotate the relative position into the parent body's coordinate system
    rel_pos = (R.from_euler("z", 90, degrees=True) * parent_rot).apply(
        np.array([rel_pos_world[0], -rel_pos_world[1], 0.0])
    )[
        :2
    ]  # Transform to parent frame

    # Compute the relative heading (yaw)
    parent_yaw = R.from_quat(parent_quat[[1, 2, 3, 0]]).as_euler("xyz", degrees=False)[
        2
    ]  # Extract parent yaw
    rel_heading = heading - parent_yaw

    # Update the qpos values for the joints
    forward_joint_idx = env.sim.model.joint_name2id("mobilebase0_joint_mobile_forward")
    side_joint_idx = env.sim.model.joint_name2id("mobilebase0_joint_mobile_side")
    yaw_joint_idx = env.sim.model.joint_name2id("mobilebase0_joint_mobile_yaw")

    env.sim.data.qpos[forward_joint_idx] = rel_pos[
        0
    ]  # Relative y position (forward joint)
    env.sim.data.qpos[side_joint_idx] = rel_pos[1]  # Relative x position (side joint)
    env.sim.data.qpos[yaw_joint_idx] = rel_heading  # Relative yaw angle (yaw joint)

    # Update the simulation state
    env.sim.forward()
    env.sim.step()

    realized_base_pose = get_base_pose(env, unwrapped=True)
    if not np.allclose(realized_base_pose, xy_heading, atol=1e-3):
        f"\033[1;31m[env_utils.py] Requested robot pose {xy_heading} is not met by realized robot position {realized_base_pose}\033[0m"


if __name__ == "__main__":
    # select random task to run rollouts for
    env_name = np.random.choice(
        list(SINGLE_STAGE_TASK_DATASETS) + list(MULTI_STAGE_TASK_DATASETS)
    )
    env = create_eval_env(env_name=env_name)
    info = run_random_rollouts(
        env, num_rollouts=3, num_steps=100, video_path="/tmp/test.mp4"
    )
