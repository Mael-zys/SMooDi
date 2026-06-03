import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
from natsort import natsorted
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualize.simplify_loc2rot import joints2smpl  # noqa: E402


GENERATED_SUFFIXES = (
    "_mesh.npy",
    "_mesh.npz",
    "_smpl.npz",
    "_smpl_params.npy",
    "_rot.npy",
)
GENERATED_NAMES = {"results.npy"}


def reverse_transpose(array):
    return array.transpose(*np.arange(array.ndim - 1, -1, -1))


def axis_angle_yup_to_zup(axis_angle, rotate_angle=90):
    rotation = R.from_rotvec(np.asarray(axis_angle))
    align = R.from_euler("x", rotate_angle, degrees=True)
    return (align * rotation).as_rotvec()


def yup_to_zup(joint_pos):
    trans_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    return np.dot(joint_pos, trans_matrix)


def vertices_to_renderer_coords(vertices, joints, opt_joints):
    vertices = np.array(vertices, copy=True)
    offset = joints[0, 0] - opt_joints[0, 0]
    vertices += offset

    x, mz, my = reverse_transpose(vertices)
    return reverse_transpose(np.stack((x, -my, mz), axis=0))


def is_generated_file(path):
    name = path.name
    return name in GENERATED_NAMES or any(
        name.endswith(suffix) for suffix in GENERATED_SUFFIXES
    )


def load_joints(path):
    try:
        array = np.load(path, allow_pickle=True)
    except Exception as exc:
        return None, f"load failed: {exc}"

    if not isinstance(array, np.ndarray):
        return None, f"expected ndarray, got {type(array).__name__}"
    if array.dtype == object:
        return None, "object arrays are not supported by this script"
    if array.ndim != 3 or array.shape[1:] != (22, 3):
        return None, f"expected shape (T, 22, 3), got {array.shape}"
    if array.shape[0] == 0:
        return None, "empty motion"

    return array.astype(np.float32, copy=False), None


def find_motion_files(input_dir):
    paths = [p for p in Path(input_dir).rglob("*.npy") if not is_generated_file(p)]
    return natsorted(paths, key=lambda p: str(p))


def get_converter(cache, num_frames, args):
    if num_frames not in cache:
        converter = joints2smpl(
            num_frames=num_frames,
            device_id=args.device,
            cuda=not args.cpu,
            num_smplify_iters=(
                args.num_smplify_iters
                if args.num_smplify_iters is not None
                else 150
            ),
        )
        cache[num_frames] = converter
    return cache[num_frames]


def save_smpl_npz(path, opt_dict, fps, text):
    poses = np.array(opt_dict["poses"], copy=True)
    poses[:, :3] = axis_angle_yup_to_zup(poses[:, :3], 90)

    np.savez(
        path,
        poses=poses,
        trans=yup_to_zup(opt_dict["trans"]),
        betas=opt_dict["betas"],
        num_betas=10,
        gender="neutral",
        mocap_frame_rate=fps,
        text=text,
    )


def render_mesh(vertices, output_path, fps):
    from renderer.humor import HumorRenderer

    cleanup_generated_frames(output_path)
    renderer = HumorRenderer(fps=fps)
    renderer(
        vertices=vertices,
        output=str(output_path),
        cam_rot=R.from_euler("x", 90, degrees=True).as_matrix(),
        cam_offset=[0.0, -2.2, 0.9],
        point_rad=0.10,
        put_ground=False,
    )


def cleanup_generated_frames(output_path):
    frame_dir = output_path.with_suffix("")
    if not frame_dir.is_dir():
        return

    entries = list(frame_dir.iterdir())
    if not entries:
        frame_dir.rmdir()
        return
    if all(
        entry.is_file()
        and entry.name.startswith("frame_")
        and entry.suffix.lower() == ".png"
        for entry in entries
    ):
        shutil.rmtree(frame_dir)


def process_file(path, converter_cache, args):
    joints, reason = load_joints(path)
    if reason is not None:
        print(f"[skip] {path}: {reason}")
        return "invalid"

    smpl_path = path.with_name(f"{path.stem}_smpl.npz")
    mesh_path = path.with_name(f"{path.stem}_mesh.npz")
    video_path = path.with_name(f"{path.stem}_mesh.mp4")

    need_fit = args.overwrite or not smpl_path.exists() or not mesh_path.exists()
    need_render = not args.no_render and (args.overwrite or not video_path.exists())

    if not need_fit and not need_render:
        print(f"[skip] {path}: outputs already exist")
        return "skipped"

    if need_fit:
        converter = get_converter(converter_cache, joints.shape[0], args)
        opt_dict = converter.joint2smpl_amass(joints.copy())

        vertices = vertices_to_renderer_coords(
            opt_dict["new_opt_vertices"],
            joints,
            opt_dict["new_opt_joints"],
        )
        save_smpl_npz(smpl_path, opt_dict, args.fps, path.stem)
        np.savez(mesh_path, vertices=vertices)
        print(f"[save] {smpl_path}")
        print(f"[save] {mesh_path}")
    else:
        vertices = np.load(mesh_path)["vertices"]

    if need_render:
        render_mesh(vertices, video_path, args.fps)
        cleanup_generated_frames(video_path)
        print(f"[save] {video_path}")

    return "processed"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert HumanML3D joints npy files to SMPL npz and mesh videos."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--num_smplify_iters", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Re-run with --cpu to fit on CPU.")

    motion_files = find_motion_files(args.input_dir)
    print(f"Found {len(motion_files)} candidate npy files under {args.input_dir}")
    converter_cache = {}
    processed = 0
    valid_seen = 0
    for path in motion_files:
        if args.max_files is not None and valid_seen >= args.max_files:
            break

        status = process_file(path, converter_cache, args)
        if status != "invalid":
            valid_seen += 1
        if status == "processed":
            processed += 1

    print(f"Done. Processed {processed}/{valid_seen} valid motion files.")


if __name__ == "__main__":
    main()
