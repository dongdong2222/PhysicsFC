import sys
sys.path.append('/workspace/CALM/fbxbinding/build/Distrib/site-packages/fbx/')

from argparse import ArgumentParser

import numpy as np
import torch

import fbx
import FbxCommon

from poselib.core.rotation3d import (
    euclidean_to_transform,
    transform_rotation,
    transform_translation,
)
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from poselib.skeleton.backend.fbx.fbx_backend import _get_skeleton, _recursive_to_list


def _find_node_by_name(root_node, name):
    stack = [root_node]
    while stack:
        node = stack.pop(0)
        if node.GetName() == name:
            return node
        for child_index in range(node.GetChildCount()):
            stack.append(node.GetChild(child_index))
    return None


def fbx_tpose_saver(fbx_file_path, root_joint_name, output_path=None):
    """
    Extracts the static (non-animated) local transform of every joint in an fbx file
    and saves it as a single-frame SkeletonState npy.

    Some fbx rigs (e.g. 3ds Max Biped exports) don't store a separate FbxPose
    bind-pose object, but the static LclTranslation/LclRotation/LclScaling
    property values (as opposed to their animated curves) still hold the rig's
    original rest/T-pose. Calling EvaluateLocalTransform() with no time argument
    reads those static properties instead of evaluating the animation curve.
    """
    manager, scene = FbxCommon.InitializeSdkObjects()
    FbxCommon.LoadScene(manager, scene, fbx_file_path)

    root_joint = _find_node_by_name(scene.GetRootNode(), root_joint_name)
    if root_joint is None:
        raise RuntimeError(
            "Could not find a node named '{}' in {}".format(root_joint_name, fbx_file_path)
        )

    joint_list, joint_names, parents = _get_skeleton(root_joint)

    local_transforms = []
    for joint in joint_list:
        arr = np.array(_recursive_to_list(joint.EvaluateLocalTransform()))
        scales = np.array(_recursive_to_list(joint.EvaluateLocalScaling()))
        if not np.allclose(scales[0:3], scales[0]):
            raise ValueError(
                "Different X, Y and Z scaling on joint '{}'. Unsure how this should be "
                "handled.".format(joint.GetName())
            )
        arr /= scales[0]
        arr[3, 3] = 1.0
        local_transforms.append(arr)

    # add a leading frame axis of size 1, matching fbx_to_npy's per-frame array shape
    local_transforms = np.array([local_transforms])

    local_transform = euclidean_to_transform(
        transformation_matrix=torch.from_numpy(
            np.swapaxes(local_transforms, -1, -2)
        ).float()
    )
    local_rotation = transform_rotation(local_transform)[0]
    local_translation = transform_translation(local_transform)[0]
    root_translation = local_translation[0]

    joint_parents = torch.from_numpy(np.array(parents)).int()
    skeleton_tree = SkeletonTree(joint_names, joint_parents, local_translation)

    skeleton_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree, r=local_rotation, t=root_translation, is_local=True
    )

    if output_path is None:
        output_path = fbx_file_path.rsplit(".fbx", 1)[0] + "_tpose.npy"
    skeleton_state.to_file(output_path)
    print(output_path, "saved")
    return output_path


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Extract a T-pose SkeletonState from an fbx file's static joint transforms"
    )
    parser.add_argument("fbx_file_path", type=str)
    parser.add_argument("root_joint_name", type=str)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    fbx_tpose_saver(args.fbx_file_path, args.root_joint_name, args.output_path)
