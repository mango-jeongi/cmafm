"""Browser UI for paired CMAFM-YOLO TensorRT inference on Jetson."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.infer_tensorrt import (
    DEFAULT_BENCHMARK_ITERATIONS,
    DEFAULT_WARMUP_ITERATIONS,
    TensorRTSession,
)
from src.postprocess import DEFAULT_NAMES, draw_detections, non_max_suppression
from src.preprocess import _thermal_to_u8, preprocess_pair, scale_boxes


DEPLOY_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = DEPLOY_DIR / "artifacts" / "cmafm_yolo_640_fp16.engine"
VIDEOS_DIR = DEPLOY_DIR.parent / "videos"
DEMO_RGB_VIDEO = VIDEOS_DIR / "flir_v1_rgb.mp4"
DEMO_THERMAL_VIDEO = VIDEOS_DIR / "flir_v1_thermal.mp4"


st.set_page_config(
    page_title="CMAFM Jetson Detection",
    page_icon="🎯",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading TensorRT engine on the Jetson GPU...")
def load_session(engine_path: str) -> TensorRTSession:
    return TensorRTSession(Path(engine_path))


def decode_rgb(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The RGB upload is not a readable image.")
    return image


def decode_thermal(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("The thermal upload is not a readable image.")
    if image.ndim == 3:
        conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        image = cv2.cvtColor(image, conversion)
    return _thermal_to_u8(image)


def detection_records(detections: np.ndarray) -> list[dict[str, object]]:
    records = []
    for row in detections:
        class_id = int(row[5])
        records.append(
            {
                "class": (
                    DEFAULT_NAMES[class_id]
                    if 0 <= class_id < len(DEFAULT_NAMES)
                    else f"class_{class_id}"
                ),
                "confidence": round(float(row[4]), 4),
                "x1": round(float(row[0]), 1),
                "y1": round(float(row[1]), 1),
                "x2": round(float(row[2]), 1),
                "y2": round(float(row[3]), 1),
            }
        )
    return records


def thermal_frame_to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        conversion = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        frame = cv2.cvtColor(frame, conversion)
    return _thermal_to_u8(frame)


def infer_pair(
    session: TensorRTSession,
    rgb_bgr: np.ndarray,
    thermal_gray: np.ndarray,
    confidence: float,
    iou: float,
    iterations: int = 1,
    warmup_iterations: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    if rgb_bgr.shape[:2] != thermal_gray.shape[:2]:
        raise ValueError(
            "RGB and thermal dimensions must match; received "
            f"{rgb_bgr.shape[:2]} and {thermal_gray.shape[:2]}."
        )

    started = time.perf_counter()
    rgb_tensor, thermal_tensor, meta = preprocess_pair(
        rgb_bgr,
        thermal_gray,
        image_size=session.image_size,
        dtype=session.input_dtype,
    )
    preprocess_ms = (time.perf_counter() - started) * 1000

    # Warmups use the real preprocessed pair and are deliberately not timed.
    for _ in range(warmup_iterations):
        session.infer(rgb_tensor, thermal_tensor)

    timings = []
    outputs = None
    for _ in range(iterations):
        started = time.perf_counter()
        outputs = session.infer(rgb_tensor, thermal_tensor)
        timings.append((time.perf_counter() - started) * 1000)
    assert outputs is not None

    started = time.perf_counter()
    detections = non_max_suppression(
        session.prediction_output(outputs),
        confidence_threshold=confidence,
        iou_threshold=iou,
    )
    detections[:, :4] = scale_boxes(detections[:, :4], meta)
    annotated = draw_detections(rgb_bgr, detections)
    postprocess_ms = (time.perf_counter() - started) * 1000
    return annotated, detections, float(np.mean(timings)), preprocess_ms, postprocess_ms


st.title("CMAFM RGB + Thermal Detection")
st.caption("FP16 TensorRT inference — Jetson Orin Nano (edge device)")

with st.sidebar:
    st.header("Runtime")
    engine_path = st.text_input("TensorRT engine", str(DEFAULT_ENGINE))
    confidence = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    iou = st.slider("NMS IoU threshold", 0.0, 1.0, 0.45, 0.01)
    iterations = st.slider(
        "Timed iterations", 1, 200, DEFAULT_BENCHMARK_ITERATIONS
    )

engine = Path(engine_path)
if not engine.is_file():
    st.error(f"TensorRT engine not found: {engine}")
    st.stop()

try:
    session = load_session(str(engine.resolve()))
except Exception as exc:
    st.exception(exc)
    st.stop()

status_a, status_b, status_c = st.columns(3)
status_a.metric("Backend", "TensorRT FP16")
status_b.metric("Input", f"{session.image_size} × {session.image_size}")
status_c.metric("Engine size", f"{engine.stat().st_size / (1024 ** 2):.1f} MiB")

image_tab, video_tab = st.tabs(["Image pair", "Paired video"])

with image_tab:
    rgb_file, thermal_file = st.columns(2)
    with rgb_file:
        rgb_upload = st.file_uploader(
            "Aligned RGB frame",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            key="image_rgb",
        )
    with thermal_file:
        thermal_upload = st.file_uploader(
            "Aligned thermal frame",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            key="image_thermal",
        )

    if rgb_upload is None or thermal_upload is None:
        st.info("Upload one spatially aligned RGB/thermal image pair to run CMAFM.")
    else:
        try:
            rgb_bgr = decode_rgb(rgb_upload.getvalue())
            thermal_gray = decode_thermal(thermal_upload.getvalue())
            annotated, detections, mean_ms, preprocess_ms, postprocess_ms = infer_pair(
                session,
                rgb_bgr,
                thermal_gray,
                confidence,
                iou,
                iterations,
                DEFAULT_WARMUP_ITERATIONS,
            )
        except Exception as exc:
            st.exception(exc)
        else:
            metric_a, metric_b, metric_c, metric_d = st.columns(4)
            metric_a.metric("Detections", len(detections))
            metric_b.metric("TensorRT inference", f"{mean_ms:.2f} ms")
            metric_c.metric("Throughput", f"{1000 / mean_ms:.1f} FPS")
            metric_d.metric(
                "Pre / post", f"{preprocess_ms:.1f} / {postprocess_ms:.1f} ms"
            )

            view_a, view_b, view_c = st.columns(3)
            with view_a:
                st.subheader("RGB")
                st.image(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with view_b:
                st.subheader("Thermal")
                st.image(thermal_gray, clamp=True, use_container_width=True)
            with view_c:
                st.subheader("CMAFM detections")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            records = detection_records(detections)
            st.subheader("Detection details")
            if records:
                st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
            else:
                st.warning("No detections passed the selected confidence threshold.")

            encoded_ok, encoded = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            download_a, download_b = st.columns(2)
            if encoded_ok:
                download_a.download_button(
                    "Download annotated image",
                    encoded.tobytes(),
                    file_name="cmafm_detection.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )
            download_b.download_button(
                "Download detection JSON",
                json.dumps(records, indent=2),
                file_name="cmafm_detection.json",
                mime="application/json",
                use_container_width=True,
            )

with video_tab:
    st.subheader("Synchronized RGB + thermal video")
    source = st.radio(
        "Video source",
        ["Jetson demo pair", "Upload MP4 pair"],
        horizontal=True,
    )

    rgb_video_upload = None
    thermal_video_upload = None
    if source == "Jetson demo pair":
        st.code(f"RGB: {DEMO_RGB_VIDEO}\nThermal: {DEMO_THERMAL_VIDEO}")
        source_ready = DEMO_RGB_VIDEO.is_file() and DEMO_THERMAL_VIDEO.is_file()
        if not source_ready:
            st.error("The demo video pair was not found on the Jetson.")
    else:
        upload_a, upload_b = st.columns(2)
        with upload_a:
            rgb_video_upload = st.file_uploader(
                "RGB video", type=["mp4"], key="video_rgb"
            )
        with upload_b:
            thermal_video_upload = st.file_uploader(
                "Thermal video", type=["mp4"], key="video_thermal"
            )
        source_ready = rgb_video_upload is not None and thermal_video_upload is not None

    option_a, option_b = st.columns(2)
    with option_a:
        max_processed_frames = st.number_input(
            "Maximum processed frames (0 = entire video)",
            min_value=0,
            max_value=10000,
            value=DEFAULT_BENCHMARK_ITERATIONS,
            step=10,
        )
    with option_b:
        frame_skip = st.number_input(
            "Process every Nth frame",
            min_value=1,
            max_value=30,
            value=1,
            step=1,
        )

    if st.button(
        "Run paired video detection",
        type="primary",
        disabled=not source_ready,
        use_container_width=True,
    ):
        uploaded_temp_paths: list[Path] = []
        output_path: Path | None = None
        cap_rgb = None
        cap_thermal = None
        writer = None
        try:
            if source == "Jetson demo pair":
                rgb_path = DEMO_RGB_VIDEO
                thermal_path = DEMO_THERMAL_VIDEO
            else:
                assert rgb_video_upload is not None and thermal_video_upload is not None
                for upload in (rgb_video_upload, thermal_video_upload):
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
                        handle.write(upload.getvalue())
                        uploaded_temp_paths.append(Path(handle.name))
                rgb_path, thermal_path = uploaded_temp_paths

            cap_rgb = cv2.VideoCapture(str(rgb_path))
            cap_thermal = cv2.VideoCapture(str(thermal_path))
            if not cap_rgb.isOpened() or not cap_thermal.isOpened():
                raise RuntimeError("OpenCV could not open one or both MP4 files.")

            rgb_count = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
            thermal_count = int(cap_thermal.get(cv2.CAP_PROP_FRAME_COUNT))
            rgb_fps = cap_rgb.get(cv2.CAP_PROP_FPS) or 30.0
            thermal_fps = cap_thermal.get(cv2.CAP_PROP_FPS) or rgb_fps
            width = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if rgb_count != thermal_count:
                raise ValueError(
                    f"Frame counts differ: RGB={rgb_count}, thermal={thermal_count}."
                )
            if abs(rgb_fps - thermal_fps) > 0.01:
                raise ValueError(f"Frame rates differ: RGB={rgb_fps}, thermal={thermal_fps}.")

            available_processed = (rgb_count + int(frame_skip) - 1) // int(frame_skip)
            target_processed = (
                available_processed
                if int(max_processed_frames) == 0
                else min(available_processed, int(max_processed_frames))
            )
            output_fps = rgb_fps / int(frame_skip)
            artifacts_dir = DEPLOY_DIR / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix="cmafm_video_", suffix=".mp4", dir=artifacts_dir, delete=False
            ) as output_handle:
                output_path = Path(output_handle.name)

            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"avc1"),
                output_fps,
                (width, height),
            )
            if not writer.isOpened():
                writer.release()
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    output_fps,
                    (width, height),
                )
            if not writer.isOpened():
                raise RuntimeError("The Jetson could not create the output MP4.")

            progress = st.progress(0.0, text="Starting paired inference...")
            preview_a, preview_b, preview_c = st.columns(3)
            preview_rgb = preview_a.empty()
            preview_thermal = preview_b.empty()
            preview_detection = preview_c.empty()
            inference_times = []
            frame_log = []
            total_detections = 0
            read_index = 0
            processed = 0

            while processed < target_processed:
                rgb_ok, rgb_frame = cap_rgb.read()
                thermal_ok, thermal_frame = cap_thermal.read()
                if not rgb_ok or not thermal_ok:
                    break
                if read_index % int(frame_skip):
                    read_index += 1
                    continue

                thermal_gray = thermal_frame_to_gray(thermal_frame)
                annotated, detections, inference_ms, _, _ = infer_pair(
                    session,
                    rgb_frame,
                    thermal_gray,
                    confidence,
                    iou,
                    warmup_iterations=(
                        DEFAULT_WARMUP_ITERATIONS if processed == 0 else 0
                    ),
                )
                writer.write(annotated)
                inference_times.append(inference_ms)
                total_detections += len(detections)
                frame_log.append(
                    {
                        "source_frame": read_index,
                        "detections": len(detections),
                        "inference_ms": round(inference_ms, 3),
                    }
                )
                processed += 1
                read_index += 1

                progress.progress(
                    processed / target_processed,
                    text=f"Processed {processed}/{target_processed} frames",
                )
                if processed == 1 or processed % 10 == 0 or processed == target_processed:
                    preview_rgb.image(
                        cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB),
                        caption="RGB",
                        use_container_width=True,
                    )
                    preview_thermal.image(
                        thermal_gray,
                        caption="Thermal",
                        clamp=True,
                        use_container_width=True,
                    )
                    preview_detection.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption="CMAFM detections",
                        use_container_width=True,
                    )

            writer.release()
            writer = None
            if processed == 0:
                raise RuntimeError("No synchronized frames were decoded.")
            output_bytes = output_path.read_bytes()
            mean_inference_ms = float(np.mean(inference_times))
            st.session_state.video_result = {
                "bytes": output_bytes,
                "processed": processed,
                "detections": total_detections,
                "mean_inference_ms": mean_inference_ms,
                "output_fps": output_fps,
                "log": frame_log,
            }
            progress.progress(1.0, text="Paired video detection complete")
        except Exception as exc:
            st.exception(exc)
        finally:
            if cap_rgb is not None:
                cap_rgb.release()
            if cap_thermal is not None:
                cap_thermal.release()
            if writer is not None:
                writer.release()
            for path in uploaded_temp_paths:
                path.unlink(missing_ok=True)
            if output_path is not None:
                output_path.unlink(missing_ok=True)

    result = st.session_state.get("video_result")
    if result:
        st.subheader("Processed CMAFM video")
        result_a, result_b, result_c, result_d = st.columns(4)
        result_a.metric("Processed frames", result["processed"])
        result_b.metric("Total detections", result["detections"])
        result_c.metric("Mean TensorRT latency", f"{result['mean_inference_ms']:.2f} ms")
        result_d.metric("Engine throughput", f"{1000 / result['mean_inference_ms']:.1f} FPS")
        st.video(result["bytes"], format="video/mp4")
        st.download_button(
            "Download processed MP4",
            result["bytes"],
            file_name="cmafm_tensorRT_detection.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
        with st.expander("Frame-by-frame measurements"):
            st.dataframe(pd.DataFrame(result["log"]), use_container_width=True, hide_index=True)
