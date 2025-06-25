from hamer_helper import HamerHelper

hamer_helper = HamerHelper()



class SingleHandHamerOutputWrtCamera(TypedDict):
    """Hand outputs with respect to the camera frame. For use in pickle files."""

    verts: np.ndarray
    keypoints_3d: np.ndarray
    mano_hand_pose: np.ndarray
    mano_hand_betas: np.ndarray
    mano_hand_global_orient: np.ndarray

pbar = tqdm(range(num_images))
for i in pbar:
    image_data, image_data_record = provider.get_image_data_by_index(
        rgb_stream_id, i
    )
    
    undistorted_image = calibration.distort_by_calibration(
        image_data.to_numpy_array(), pinhole, camera_calib
    )

    hamer_out_left, hamer_out_right = hamer_helper.look_for_hands(
        undistorted_image,
        focal_length=450,
    )

    timestamp_ns = image_data_record.capture_timestamp_ns


    # Dict from capture timestamp in nanoseconds to fields we care about.
    detections_left_wrt_cam: dict[int, SingleHandHamerOutputWrtCamera | None] = {}
    detections_right_wrt_cam: dict[int, SingleHandHamerOutputWrtCamera | None] = {}


    if hamer_out_left is None:
        detections_left_wrt_cam[timestamp_ns] = None
    else:
        detections_left_wrt_cam[timestamp_ns] = {
            "verts": hamer_out_left["verts"],
            "keypoints_3d": hamer_out_left["keypoints_3d"],
            "mano_hand_pose": hamer_out_left["mano_hand_pose"],
            "mano_hand_betas": hamer_out_left["mano_hand_betas"],
            "mano_hand_global_orient": hamer_out_left["mano_hand_global_orient"],
        }

    if hamer_out_right is None:
        detections_right_wrt_cam[timestamp_ns] = None
    else:
        detections_right_wrt_cam[timestamp_ns] = {
            "verts": hamer_out_right["verts"],
            "keypoints_3d": hamer_out_right["keypoints_3d"],
            "mano_hand_pose": hamer_out_right["mano_hand_pose"],
            "mano_hand_betas": hamer_out_right["mano_hand_betas"],
            "mano_hand_global_orient": hamer_out_right["mano_hand_global_orient"],
        }