# # scripts/annote_data.py
# import os
# import numpy as np
# from PIL import Image
# import torchvision.transforms as T
# import cv2
# from abc import ABC, abstractmethod
# import sys
# import yaml
# from omegaconf import OmegaConf

# sys.path.append("/home/voanhnhat/Documents/PROJECT_DETECT_OBJECT/sam2")
# from sam2.build_sam import build_sam2
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# class SAM2Model:
#     def __init__(self, checkpoint_path, config_path):
#         # Use absolute paths for safety
#         checkpoint_path = os.path.abspath(checkpoint_path)
#         config_path = os.path.abspath(config_path)
        
#         # Load the YAML config manually to fix targets
#         with open(config_path, 'r') as f:
#             raw_cfg = yaml.safe_load(f)
        
#         # Fix the _target_ paths by removing extra 'sam2.'
#         def fix_targets(cfg_dict):
#             for key, value in cfg_dict.items():
#                 if isinstance(value, dict):
#                     fix_targets(value)
#                 elif key == '_target_' and isinstance(value, str) and value.startswith('sam2.sam2.'):
#                     cfg_dict[key] = value.replace('sam2.sam2.', 'sam2.')
        
#         fix_targets(raw_cfg)
        
#         # Save the fixed config to a temporary YAML file
#         temp_config_path = config_path + '.fixed.yaml'
#         with open(temp_config_path, 'w') as f:
#             yaml.safe_dump(raw_cfg, f)
        
#         # Now build the model using the fixed config path (string)
#         self.model = build_sam2(temp_config_path, checkpoint_path)
        
#         # Clean up the temporary file
#         os.remove(temp_config_path)
        
#         self.mask_generator = SAM2AutomaticMaskGenerator(self.model)

#     def generate_masks(self, image):
#         return self.mask_generator.generate(image)  # Sinh mask cho 1 ảnh


# class BaseAnnotator(ABC):
#     def __init__(self, model, input_path, output_dir):
#         self.model = model  # Mô hình sam2
#         self.input_path = input_path  # Đường dẫn dữ liệu vào
#         self.output_dir = output_dir  # Nơi lưu annote
#         # Đảm bảo thư mục tồn tại trước khi lưu file.
#         os.makedirs(self.output_dir, exist_ok=True)

#     def load_model(self):
#         return self.model  # Nạp và khởi tạo model

#     # Hàm trừu tượng.
#     @abstractmethod
#     def annotate(self):
#         pass


# class ImageAnnotator(BaseAnnotator):
#     def __init__(self, model, input_path, output_dir, image_size=(640, 640)):
#         super().__init__(model, input_path, output_dir)
#         self.image_size = image_size

#     def preprocess_image(self):
#         # Đọc ảnh và chuyển sang RGB
#         img = Image.open(self.input_path).convert("RGB")
#         # resize ảnh về đúng kích thước, chuẩn hóa
#         resize = T.Resize(self.image_size)
#         img = resize(img)
#         return np.array(img)  # Output: numpy array(h, w, 3(bgr))

#     def annotate(self):
#         """Masks là vùng bôi lên đối tượng mà bạn cần chọn."""
#         img = self.preprocess_image()
#         masks = self.model.generate_masks(img)  # Trả về dictionary.
#         # Kiểm tra có masks nào đang tồn tại không.
#         if masks is None or len(masks) == 0:
#             print("No masks found for image:", self.input_path)
#             return False
#         best_score = None  # Độ tin cậy cao nhất
#         best_masks_array = None  # Masks có độ tin cậy cao nhất
#         # Trả về chỉ số index và thông tin masks.
#         for index, m in enumerate(masks):
#             score = m.get("score", 0.0)
#             segmentation = m.get("segmentation", None)

#             if segmentation is None:
#                 print(f"Masks {index} no segmentation")
#                 continue
#             # Chuyển từ masks nhị phân sang numpy array 8-bit
#             segmentation_array = np.array(segmentation, dtype=np.uint8)

#             if best_score is None or score > best_score:
#                 best_score = score
#                 best_masks_array = segmentation_array
#         if best_masks_array is None:
#             print("No valid masks after review")
#             return False
#         # Lưu mask ra file
#         file_name = os.path.splitext(os.path.basename(self.input_path))[0]
#         mask_path = os.path.join(self.output_dir, f"{file_name}_mask.png")
#         cv2.imwrite(mask_path, best_masks_array * 255)
#         print(f"Saved mask for {self.input_path} in {mask_path}")
#         return True


# class VideoAnnotator(BaseAnnotator):
#     def __init__(self, model, input_path, output_dir, frame_step=30, image_size=(640, 640)):
#         super().__init__(model, input_path, output_dir)
#         self.image_size = image_size
#         self.frame_step = frame_step

#     def annotate(self):
#         cap = cv2.VideoCapture(self.input_path)
#         frame_count = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             if frame_count % self.frame_step == 0:  # annotate mỗi N frame
#                 # Chuyển frame sang RGB.
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 # Resize về kích thước chuẩn.
#                 frame = cv2.resize(frame, self.image_size)
#                 masks = self.model.generate_masks(frame)
#                 if masks and len(masks) > 0:
#                     best_score = -1
#                     best_masks_inf = None
#                     for m in masks:
#                         score = m.get("score", 0.0)
#                         if score > best_score:
#                             best_score = score
#                             best_masks_inf = m
#                     if best_masks_inf and "segmentation" in best_masks_inf:
#                         segmentation_array = np.array(
#                             best_masks_inf["segmentation"], dtype=np.uint8)
#                         file_name = f"frame_{frame_count}_mask.png"
#                         mask_path = os.path.join(self.output_dir, file_name)
#                         cv2.imwrite(mask_path, segmentation_array * 255)
#                         print(f"Mask saved: {mask_path}")
#                     else:
#                         print(f"Frame {frame_count}: no valid segmentation")
#             frame_count += 1
#         cap.release()
#         print(f"Complete annotate video {self.input_path}")
#         return True


# class AnnotationPipeline:
#     def __init__(self, checkpoint_path, config_path, raw_dir, annotated_dir):
#         self.model = SAM2Model(checkpoint_path, config_path)
#         self.raw_dir = raw_dir
#         self.annotated_dir = annotated_dir
#         # Đảm bảo thư mục tồn tại khi lưu file.
#         os.makedirs(self.annotated_dir, exist_ok=True)

#     def run(self):
#         # Duyệt toàn bộ folder raw_dir, annotate ảnh và video.
#         for category in os.listdir(self.raw_dir):
#             category_path = os.path.join(self.raw_dir, category)
#             if not os.path.isdir(category_path):
#                 continue
#             output_path = os.path.join(self.annotated_dir, category)
#             os.makedirs(output_path, exist_ok=True)
#             for file_name in os.listdir(category_path):
#                 file_path = os.path.join(category_path, file_name)
#                 if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
#                     annotator = ImageAnnotator(
#                         self.model, file_path, output_path)
#                     annotator.annotate()
#                 elif file_name.lower().endswith(('.mp4', '.avi', '.mov')):
#                     annotator = VideoAnnotator(
#                         self.model, file_path, output_path)
#                     annotator.annotate()
#         print("Annotation complete! Result in:", self.annotated_dir)


# # main
# if __name__ == "__main__":
#     pipeline = AnnotationPipeline(
#         checkpoint_path="/home/voanhnhat/Documents/PROJECT_DETECT_OBJECT/checkpoints/sam2.1_hiera_large.pt",
#         config_path="/home/voanhnhat/Documents/PROJECT_DETECT_OBJECT/configs/sam2.1/sam2.1_hiera_l.yaml",
#         raw_dir="/home/voanhnhat/Documents/PROJECT_DETECT_OBJECT/data/raw",
#         annotated_dir="/home/voanhnhat/Documents/PROJECT_DETECT_OBJECT/data/annotated"
#     )
#     pipeline.run()



# scripts/annote_data.py
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
import cv2
from abc import ABC, abstractmethod
import sys
import yaml
from omegaconf import OmegaConf

sys.path.append("/home/voanhnhat/Documents/PROJECT_DETECT_OBJECT/sam2")
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


class SAM2Model:
    def __init__(self, checkpoint_path, config_path):
        # Use absolute paths for safety
        checkpoint_path = os.path.abspath(checkpoint_path)
        config_path = os.path.abspath(config_path)
        
        # Load the YAML config manually
        with open(config_path, 'r') as f:
            raw_cfg = yaml.safe_load(f)
        
        # Fix the _target_ paths by removing extra 'sam2.'
        def fix_targets(cfg_dict):
            for key, value in cfg_dict.items():
                if isinstance(value, dict):
                    fix_targets(value)
                elif key == '_target_' and isinstance(value, str) and value.startswith('sam2.sam2.'):
                    cfg_dict[key] = value.replace('sam2.sam2.', 'sam2.')
        
        fix_targets(raw_cfg)
        
        # Save the fixed config to a temporary YAML file
        temp_config_path = config_path + '.fixed.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.safe_dump(raw_cfg, f)
        
        # Now build the model using the fixed config path (string)
        self.model = build_sam2(temp_config_path, checkpoint_path)
        
        # Clean up the temporary file
        os.remove(temp_config_path)
        
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model)

    def generate_masks(self, image):
        return self.mask_generator.generate(image)  # Sinh mask cho 1 ảnh


class BaseAnnotator(ABC):
    def __init__(self, model, input_path, output_dir):
        self.model = model  # Mô hình sam2
        self.input_path = input_path  # Đường dẫn dữ liệu vào
        self.output_dir = output_dir  # Nơi lưu annote
        os.makedirs(self.output_dir, exist_ok=True)

    def load_model(self):
        return self.model  # Nạp và khởi tạo model

    # Hàm trừu tượng.
    @abstractmethod
    def annotate(self):
        pass


class ImageAnnotator(BaseAnnotator):
    def __init__(self, model, input_path, output_dir, image_size=(640, 640)):
        super().__init__(model, input_path, output_dir)
        self.image_size = image_size

    def preprocess_image(self):
        # Đọc ảnh và chuyển sang RGB
        img = Image.open(self.input_path).convert("RGB")
        # resize ảnh về đúng kích thước, chuẩn hóa
        resize = T.Resize(self.image_size)
        img = resize(img)
        return np.array(img)  # Output: numpy array(h, w, 3(bgr))

    def annotate(self):
        """Masks là vùng bôi lên đối tượng mà bạn cần chọn."""
        img = self.preprocess_image()
        masks = self.model.generate_masks(img)  # Trả về dictionary.
        # Kiểm tra có masks nào đang tồn tại không.
        if masks is None or len(masks) == 0:
            print("No masks found for image:", self.input_path)
            return False
        best_score = None  # Độ tin cậy cao nhất
        best_masks_array = None  # Masks có độ tin cậy cao nhất
        # Trả về chỉ số index và thông tin masks.
        for index, m in enumerate(masks):
            score = m.get("score", 0.0)
            segmentation = m.get("segmentation", None)

            if segmentation is None:
                print(f"Masks {index} no segmentation")
                continue
            # Chuyển từ masks nhị phân sang numpy array 8-bit
            segmentation_array = np.array(segmentation, dtype=np.uint8)

            if best_score is None or score > best_score:
                best_score = score
                best_masks_array = segmentation_array
        if best_masks_array is None:
            print("No valid masks after review")
            return False
        # Lưu mask ra file
        file_name = os.path.splitext(os.path.basename(self.input_path))[0]
        mask_path = os.path.join(self.output_dir, f"{file_name}_mask.png")
        cv2.imwrite(mask_path, best_masks_array * 255)
        print(f"Saved mask for {self.input_path} in {mask_path}")
        return True


class VideoAnnotator(BaseAnnotator):
    def __init__(self, model, input_path, output_dir, frame_step=30, image_size=(640, 640)):
        super().__init__(model, input_path, output_dir)
        self.image_size = image_size
        self.frame_step = frame_step

    def annotate(self):
        cap = cv2.VideoCapture(self.input_path)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % self.frame_step == 0:  # annotate mỗi N frame
                # Chuyển frame sang RGB.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize về kích thước chuẩn.
                frame = cv2.resize(frame, self.image_size)
                masks = self.model.generate_masks(frame)
                if masks and len(masks) > 0:
                    best_score = -1
                    best_masks_inf = None
                    for m in masks:
                        score = m.get("score", 0.0)
                        if score > best_score:
                            best_score = score
                            best_masks_inf = m
                    if best_masks_inf and "segmentation" in best_masks_inf:
                        segmentation_array = np.array(
                            best_masks_inf["segmentation"], dtype=np.uint8)
                        file_name = f"frame_{frame_count}_mask.png"
                        mask_path = os.path.join(self.output_dir, file_name)
                        cv2.imwrite(mask_path, segmentation_array * 255)
                        print(f"Mask saved: {mask_path}")
                    else:
                        print(f"Frame {frame_count}: no valid segmentation")
            frame_count += 1
        cap.release()
        print(f"Complete annotate video {self.input_path}")
        return True


class AnnotationPipeline:
    def __init__(self, checkpoint_path, config_path, raw_dir, annotated_dir):
        self.model = SAM2Model(checkpoint_path, config_path)
        self.raw_dir = raw_dir
        self.annotated_dir = annotated_dir
        # Đảm bảo thư mục tồn tại khi lưu file.
        os.makedirs(self.annotated_dir, exist_ok=True)

    def run(self):
        # Duyệt recursive tất cả file trong raw_dir và thư mục con
        for root, dirs, files in os.walk(self.raw_dir):
            relative_root = os.path.relpath(root, self.raw_dir)
            output_path = os.path.join(self.annotated_dir, relative_root)
            os.makedirs(output_path, exist_ok=True)
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    annotator = ImageAnnotator(
                        self.model, file_path, output_path)
                    annotator.annotate()
                elif file_name.lower().endswith(('.mp4', '.avi', '.mov')):
                    annotator = VideoAnnotator(
                        self.model, file_path, output_path)
                    annotator.annotate()
        print("Annotation complete! Result in:", self.annotated_dir)


# main
if __name__ == "__main__":
    pipeline = AnnotationPipeline(
        checkpoint_path="/home/voanhnhat/Documents/PROJECT_DETECT_OBJECT/checkpoints/sam2.1_hiera_large.pt",
        config_path="/home/voanhnhat/Documents/PROJECT_DETECT_OBJECT/configs/sam2.1/sam2.1_hiera_l.yaml",
        raw_dir="/home/voanhnhat/Documents/PROJECT_DETECT_OBJECT/data/raw",
        annotated_dir="/home/voanhnhat/Documents/PROJECT_DETECT_OBJECT/data/annotated"
    )
    pipeline.run()