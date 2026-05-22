"""Extract frames from an orbit video and write metadata for downstream 3D steps."""

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Dict

import cv2

from src.io.frame_writer import save_frame, save_metadata
from src.io.video_reader import open_video, iter_frames
from src.preprocess.frame_filter import sharpness_laplacian, keep_frame_by_blur


def build_argparser() -> ArgumentParser:
    parser = ArgumentParser(description="Extract frames from a video for 3D reconstruction.")
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument("--out", type=Path, default=Path("data/frames"), help="Output frames directory")
    parser.add_argument("--fps", type=float, default=None, help="Target output FPS")
    parser.add_argument("--stride", type=int, default=1, help="Take every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum accepted frames")
    parser.add_argument("--resize-width", type=int, default=None, help="Optional resize width")
    parser.add_argument("--resize-height", type=int, default=None, help="Optional resize height")
    parser.add_argument("--blur-threshold", type=float, default=0.0, help="Minimum sharpness score")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Start timestamp in seconds")
    parser.add_argument("--end-sec", type=float, default=None, help="End timestamp in seconds")
    parser.add_argument(
        "--save-metadata",
        type=Path,
        default=Path("data/frames/metadata.json"),
        help="Output metadata JSON path",
    )
    return parser


def _should_take_frame(src_index: int, src_fps: float, timestamp_sec: float, args) -> bool:
    if timestamp_sec < args.start_sec:
        return False
    if args.end_sec is not None and timestamp_sec > args.end_sec:
        return False

    if args.fps is not None and src_fps > 0:
        effective_stride = max(1, int(round(src_fps / args.fps)))
    else:
        effective_stride = max(1, args.stride)

    return (src_index % effective_stride) == 0


# Frame by frame processing is used so that we can get the video timestamp for each frame that is going from the orbit video
def main() -> None:
    args = build_argparser().parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.save_metadata == Path("data/frames/metadata.json"):
        args.save_metadata = args.out / "metadata.json"

    cap = open_video(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    accepted_count = 0
    metadata: List[Dict] = []

    for src_index, timestamp_sec, frame in iter_frames(cap):
        if not _should_take_frame(src_index, src_fps, timestamp_sec, args):
            continue

        if args.resize_width and args.resize_height:
            frame = cv2.resize(frame, (args.resize_width, args.resize_height))

        sharpness_score = sharpness_laplacian(frame)
        if not keep_frame_by_blur(sharpness_score, args.blur_threshold):
            continue

        file_name = f"frame_{accepted_count:05d}.jpg"
        out_path = args.out / file_name
        save_frame(frame, out_path)

        metadata.append(
            {
                "frame_id": accepted_count,
                "src_frame_index": src_index,
                "timestamp_sec": timestamp_sec,
                "sharpness_score": sharpness_score,
                "file_name": file_name,
            }
        )

        accepted_count += 1
        if args.max_frames is not None and accepted_count >= args.max_frames:
            break

    cap.release()
    save_metadata(metadata, args.save_metadata)

    print(f"Accepted {accepted_count} frames from {args.video} into {args.out}")
    print(f"Saved metadata: {args.save_metadata}")


if __name__ == "__main__":
    main()

