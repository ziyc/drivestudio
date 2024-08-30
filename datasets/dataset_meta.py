DATASETS_CONFIG = {
    "waymo": {
        0: {
            "camera_name": "front_camera",
            "original_size": (1280, 1920),
            "egocar_visible": False
        },
        1: {
            "camera_name": "front_left_camera",
            "original_size": (1280, 1920),
            "egocar_visible": False
        },
        2: {
            "camera_name": "front_right_camera",
            "original_size": (1280, 1920),
            "egocar_visible": False
        },
        3: {
            "camera_name": "left_camera",
            "original_size": (866, 1920),
            "egocar_visible": False
        },
        4: {
            "camera_name": "right_camera",
            "original_size": (866, 1920),
            "egocar_visible": False
        },
    },
    "pandaset": {
        0: {
            "camera_name": "front_camera",
            "original_size": (1080, 1920),
            "egocar_visible": False
        },
        1: {
            "camera_name": "front_left_camera",
            "original_size": (1080, 1920),
            "egocar_visible": False
        },
        2: {
            "camera_name": "front_right_camera",
            "original_size": (1080, 1920),
            "egocar_visible": False
        },
        3: {
            "camera_name": "left_camera",
            "original_size": (1080, 1920),
            "egocar_visible": False
        },
        4: {
            "camera_name": "right_camera",
            "original_size": (1080, 1920),
            "egocar_visible": False
        },
        5: {
            "camera_name": "back_camera",
            "original_size": (1080, 1920),
            "egocar_visible": True
        },
    },
    "argoverse": {
        0: {
            "camera_name": "ring_front_center",
            "original_size": (2048, 1550),
            "egocar_visible": True
        },
        1: {
            "camera_name": "ring_front_left",
            "original_size": (1550, 2048),
            "egocar_visible": False
        },
        2: {
            "camera_name": "ring_front_right",
            "original_size": (1550, 2048),
            "egocar_visible": False
        },
        3: {
            "camera_name": "ring_side_left",
            "original_size": (1550, 2048),
            "egocar_visible": False
        },
        4: {
            "camera_name": "ring_side_right",
            "original_size": (1550, 2048),
            "egocar_visible": False
        },
        5: {
            "camera_name": "ring_rear_left",
            "original_size": (1550, 2048),
            "egocar_visible": True
        },
        6: {
            "camera_name": "ring_rear_right",
            "original_size": (1550, 2048),
            "egocar_visible": True
        },  
    },
    "nuscenes": {
        0: {
            "camera_name": "CAM_FRONT",
            "original_size": (900, 1600),
            "egocar_visible": False
        },
        1: {
            "camera_name": "CAM_FRONT_LEFT",
            "original_size": (900, 1600),
            "egocar_visible": False
        },
        2: {
            "camera_name": "CAM_FRONT_RIGHT",
            "original_size": (900, 1600),
            "egocar_visible": False
        },
        3: {
            "camera_name": "CAM_BACK_LEFT",
            "original_size": (900, 1600),
            "egocar_visible": False
        },
        4: {
            "camera_name": "CAM_BACK_RIGHT",
            "original_size": (900, 1600),
            "egocar_visible": False
        },
        5: {
            "camera_name": "CAM_BACK",
            "original_size": (900, 1600),
            "egocar_visible": True
        }
    },
    "kitti": {
        0: {
            "camera_name": "CAM_LEFT",
            "original_size": (375, 1242),
            "egocar_visible": False
        },
        1: {
            "camera_name": "CAM_RIGHT",
            "original_size": (375, 1242),
            "egocar_visible": False
        }
    },
    "nuplan": {
        0: {
            "camera_name": "CAM_F0",
            "original_size": (1080, 1920),
            "egocar_visible": False
        },
        1: {
            "camera_name": "CAM_L0",
            "original_size": (1080, 1920),
            "egocar_visible": True
        },
        2: {
            "camera_name": "CAM_R0",
            "original_size": (1080, 1920),
            "egocar_visible": True
        },
        3: {
            "camera_name": "CAM_L1",
            "original_size": (1080, 1920),
            "egocar_visible": False
        },
        4: {
            "camera_name": "CAM_R1",
            "original_size": (1080, 1920),
            "egocar_visible": False
        },
        5: {
            "camera_name": "CAM_L2",
            "original_size": (1080, 1920),
            "egocar_visible": True
        },
        6: {
            "camera_name": "CAM_R2",
            "original_size": (1080, 1920),
            "egocar_visible": True
        },
        7: {
            "camera_name": "CAM_B0",
            "original_size": (1080, 1920),
            "egocar_visible": False
        },
    },
}
