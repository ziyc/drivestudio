import sqlite3
from typing import Generator

import numpy as np
from dataclasses import dataclass
from pyquaternion import Quaternion

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType
from nuplan.common.utils.helpers import get_unique_incremental_track_id
from nuplan.database.nuplan_db.query_session import execute_many, execute_one
from nuplan.database.utils.label.utils import local2agent_type, raw_mapping

@dataclass
class SimplifiedTrackedObject:
    track_token: str
    category: str
    pose: np.array
    box_size: np.array

def _parse_tracked_object_row(row: sqlite3.Row) -> SimplifiedTrackedObject:
    """
    A convenience method to parse a TrackedObject from a sqlite3 row.
    :param row: The row from the DB query.
    :return: The parsed TrackedObject.
    """
    category_name = row["category_name"]
    pose = StateSE2(row["x"], row["y"], row["yaw"])
    oriented_box = OrientedBox(pose, width=row["width"], length=row["length"], height=row["height"])

    # These next two are globals
    label_local = raw_mapping["global2local"][category_name]
    tracked_object_type = TrackedObjectType[local2agent_type[label_local]]

    if tracked_object_type in AGENT_TYPES:
        obj =  Agent(
            tracked_object_type=tracked_object_type,
            oriented_box=oriented_box,
            velocity=StateVector2D(row["vx"], row["vy"]),
            predictions=[],  # to be filled in later
            angular_velocity=np.nan,
            metadata=SceneObjectMetadata(
                token=row["token"].hex(),
                track_token=row["track_token"].hex(),
                track_id=get_unique_incremental_track_id(str(row["track_token"].hex())),
                timestamp_us=row["timestamp"],
                category_name=category_name,
            ),
        )
    else:
        obj = StaticObject(
            tracked_object_type=tracked_object_type,
            oriented_box=oriented_box,
            metadata=SceneObjectMetadata(
                token=row["token"].hex(),
                track_token=row["track_token"].hex(),
                track_id=get_unique_incremental_track_id(str(row["track_token"].hex())),
                timestamp_us=row["timestamp"],
                category_name=category_name,
            ),
        )
    
    # object pose in 3D space
    obj_pose3d = obj.box.center.as_matrix_3d() # StateSE2 to np.array
    obj_pose3d[2, 3] = row["z"] # add z to the pose to get 3D pose
    
    # box_size is [l, w, h]
    l, w, h = obj.box.length, obj.box.width, obj.box.height
    box_size = np.array([l, w, h])
    
    return SimplifiedTrackedObject(
        category=tracked_object_type.name.lower(),
        pose=obj_pose3d,
        box_size=box_size,
        track_token=row["track_token"].hex()
    )

def get_egopose3d_for_lidarpc_token_from_db(log_file: str, token: str) -> np.array:
    """
    Get the ego pose as a StateSE2 from the db for a given lidar_pc token.
    :param log_file: The db file to query.
    :param token: The token for which to query the current state.
    :return: The current ego state, as a StateSE2 object.
    """
    query = """
        SELECT  ep.x,
                ep.y,
                ep.z,
                ep.qw,
                ep.qx,
                ep.qy,
                ep.qz
        FROM ego_pose AS ep
        INNER JOIN lidar_pc AS lp
            ON lp.ego_pose_token = ep.token
        WHERE lp.token = ?
    """

    row = execute_one(query, (bytearray.fromhex(token),), log_file)
    if row is None:
        return None

    q = Quaternion(row["qw"], row["qx"], row["qy"], row["qz"])
    rotation = q.rotation_matrix
    translation = np.array([row["x"], row["y"], row["z"]])
    
    ego_pose = np.eye(4)
    ego_pose[:3, :3] = rotation
    ego_pose[:3, 3] = translation
    
    return ego_pose

def get_tracked_objects_for_lidarpc_token_from_db(log_file: str, token: str) -> Generator[TrackedObject, None, None]:
    """
    Get all tracked objects for a given lidar_pc.
    This includes both agents and static objects.
    The values are returned in random order.

    For agents, this query will not obtain the future waypoints.
    For that, call `get_future_waypoints_for_agents_from_db()`
        with the tokens of the agents of interest.

    :param log_file: The log file to query.
    :param token: The lidar_pc token for which to obtain the objects.
    :return: The tracked objects associated with the token.
    """
    query = """
        SELECT  c.name AS category_name,
                lb.x,
                lb.y,
                lb.z,
                lb.yaw,
                lb.width,
                lb.length,
                lb.height,
                lb.vx,
                lb.vy,
                lb.token,
                lb.track_token,
                lp.timestamp
        FROM lidar_box AS lb
        INNER JOIN track AS t
            ON t.token = lb.track_token
        INNER JOIN category AS c
            ON c.token = t.category_token
        INNER JOIN lidar_pc AS lp
            ON lp.token = lb.lidar_pc_token
        WHERE lp.token = ?
    """

    for row in execute_many(query, (bytearray.fromhex(token),), log_file):
        yield _parse_tracked_object_row(row)