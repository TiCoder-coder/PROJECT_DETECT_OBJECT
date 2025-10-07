# scripts/inference_webcam.py
import os
import sys
import cv2
import torch
import numpy as np
import threading
import queue
import time
import matplotlib.pyplot as plt

# Project root (PROJECT_DETECT_OBJECT)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# ensure project root on sys.path so imports like sam2.sam2.* resolve correctly
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def resolve_path(maybe_path):
    """Resolve a path that can be absolute or relative to project root."""
    if os.path.isabs(maybe_path) and os.path.exists(maybe_path):
        return maybe_path
    # try relative to project root
    p = os.path.join(PROJECT_ROOT, maybe_path)
    if os.path.exists(p):
        return p
    # try under sam2/ (some users use configs relative to sam2 folder)
    p2 = os.path.join(PROJECT_ROOT, "sam2", maybe_path)
    if os.path.exists(p2):
        return p2
    # not found, return original so downstream code can warn
    return maybe_path


class WebcamInferencer:
    def __init__(self, model_cfg: str, checkpoint: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg_path = resolve_path(model_cfg)
        ckpt_path = resolve_path(checkpoint)

        print(f"[INFO] Using config: {cfg_path}")
        print(f"[INFO] Using checkpoint: {ckpt_path}")
        print(f"[INFO] Device: {self.device}")

        # Build SAM2 model for image segmentation
        self.sam2_model = build_sam2(
            config_file=cfg_path,
            ckpt_path=ckpt_path,
            device=self.device,
        )

        # Use AutomaticMaskGenerator with optimized parameters for speed
        self.generator = SAM2AutomaticMaskGenerator(
            model=self.sam2_model,
            points_per_side=16,  # Reduced for faster inference (default is 32)
            pred_iou_thresh=0.9,  # Higher threshold to filter more masks
            stability_score_thresh=0.92,  # Higher for fewer but more stable masks
            crop_n_layers=0,  # Disable multi-scale for speed
            min_mask_region_area=200,  # Larger to ignore small regions
        )

        print(f"[INFO] Using optimized AutomaticMaskGenerator for faster real-time segmentation.")

        # Queues for frames and masks
        self.frame_queue = queue.Queue(maxsize=5)  # Queue for frames to infer
        self.mask_queue = queue.Queue(maxsize=1)   # Queue for resulting annotations
        self.last_anns = []  # Store last annotations to reuse if inference lags
        self.stop_event = threading.Event()
        self.inference_thread = None

        # Setup matplotlib for display
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title('SAM2 Webcam Object Detection')
        self.im = None

    @torch.inference_mode()  # Optimize inference by disabling gradients
    def infer_frame(self, frame: np.ndarray) -> list:
        """Generate masks for a single frame using automatic segmentation."""
        # Resize frame for faster inference (e.g., to 640x480)
        resized_frame = cv2.resize(frame, (640, 480))
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        # Generate masks
        masks = self.generator.generate(image_rgb)
        return masks

    def inference_worker(self):
        """Thread for running inference on frames from queue."""
        skip_counter = 0
        skip_frames = 3  # Infer every 3 frames to reduce load
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)  # Wait briefly for a frame
                skip_counter += 1
                if skip_counter % skip_frames != 0:
                    continue  # Skip inference
                anns = self.infer_frame(frame)
                # Put new annotations in queue (non-blocking)
                try:
                    self.mask_queue.put_nowait(anns)
                except queue.Full:
                    pass  # Skip if queue is full (use previous)
            except queue.Empty:
                continue  # No frame available, loop again
            time.sleep(0.01)  # Small sleep to avoid CPU overload

    def run(self, cam_id: int = 0):
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera id {cam_id}")

        # Set lower resolution for webcam to reduce processing load
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Limit FPS if possible

        # Start inference thread
        self.inference_thread = threading.Thread(target=self.inference_worker)
        self.inference_thread.start()

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"[WARNING] Failed to read frame {frame_idx}. Ending stream.")
                break

            # Flip the frame horizontally to correct the mirror effect
            frame = cv2.flip(frame, 1)

            # Put frame into queue for inference (non-blocking)
            try:
                self.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass  # Queue full, skip this frame for inference

            # Get latest annotations from queue if available, else use last
            try:
                self.last_anns = self.mask_queue.get_nowait()
            except queue.Empty:
                pass  # Use previous annotations

            # Overlay masks on the frame
            for ann in self.last_anns:
                mask = ann['segmentation']  # Boolean mask (HxW)
                # Resize mask back to original frame size if needed
                mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = mask > 0
                # Apply green overlay with transparency
                overlay = frame.copy()
                overlay[mask] = (0, 255, 0)
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

                # Draw bounding box
                bbox = ann['bbox']  # [x, y, w, h]
                scale_x = frame.shape[1] / 640
                scale_y = frame.shape[0] / 480
                x, y, w, h = int(bbox[0] * scale_x), int(bbox[1] * scale_y), int(bbox[2] * scale_x), int(bbox[3] * scale_y)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display the frame using matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.im is None:
                self.im = self.ax.imshow(frame_rgb)
                self.ax.axis('off')
            else:
                self.im.set_data(frame_rgb)
            plt.draw()
            plt.pause(0.001)  # Pause to update the display

            # Check for 'q' key press (using matplotlib key event)
            if plt.waitforbuttonpress(timeout=0.001):
                if plt.gcf().canvas.key_press_handler_id:  # Simulate 'q' check, but matplotlib doesn't directly support cv2.waitKey
                    print("[INFO] Window closed or key pressed.")
                    break

            frame_idx += 1

        self.stop_event.set()
        if self.inference_thread:
            self.inference_thread.join()

        cap.release()
        plt.close(self.fig)
        print("[INFO] Webcam stream ended.")


if __name__ == "__main__":
    # Suggest using a smaller model for better performance
    # inferencer = WebcamInferencer(
    #     model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
    #     checkpoint="checkpoints/sam2.1_hiera_tiny.pt",
    # )
    inferencer = WebcamInferencer(
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        checkpoint="checkpoints/sam2.1_hiera_large.pt",
    )
    inferencer.run(cam_id=0)