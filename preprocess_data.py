import os
import json
import numpy as np
import cv2
from PIL import Image


class Annotation:
    """Class này xử lí contour, tạo annotation theo COCO"""

    def __init__(self, contour, image_id, ann_id, category_id=1):
        self.contour = contour.reshape(-1, 2)
        self.image_id = image_id
        self.ann_id = ann_id
        self.category_id = category_id

    def isVaild(self):
        """Kiểm tra contour có đưa vào hợp lệ không"""
        return self.contour.ndim == 2 and len(self.contour) >= 3

    def segmentation(self):
        # Flatten contour thành segemntation (COCO format).
        return [self.contour.flatten().tolist()]

    def area(self):
        """Tính diện tích contour"""
        return float(cv2.contourArea(self.contour))

    def bbox(self):
        """Tính bounnding box [x_min, y_min, width, height]"""
        """Cho ra mảng numpy 2D
           Lấy tất cả các giá trị ở cột 0 và 1."""
        x_coords = self.contour[:, 0]
        y_coords = self.contour[:, 1]
        """Lấy tất cả giá trị nhỏ nhất và lớn nhất theo trục hoành"""
        x_min, x_max = x_coords.min(), x_coords.max()
        """Lấy tất cả các giá trị nhỏ nhất và lớn nhất của y theo trục tung"""
        y_min, y_max = y_coords.min(), y_coords.max()
        return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

    def to_dict(self):
        """Trả về annotation dictionary theo COCO format"""
        return {
            "id": self.ann_id,
            "image_id": self.image_id,
            "category_id": self.category_id,
            "segmentation": self.segmentation,
            "area": self.area(),
            "bbox": self.bbox(),
            "iscrowd": 0
        }


class COCODataset:
    def __init__(self, description="Custom SAM2 Dataset", data_created="unknow"):
        self.data = {
            "info": {"description": description, "data_created": data_created},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "object"}]
        }

    def add_image(self, img_id, file_name, width, height):
        self.data["images"].append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

    def add_annotation(self, annotation: Annotation):
        """Kiểm tra annotation đưa vào có hợp lệ không. Nếu hợp lệ thì đưa vào danh sách"""
        if (annotation.isVaild()):
            """Chuyển annotation về dạng dictionary"""
            self.data["annotations"].append(annotation.to_dict())

    def save(self, output_path):
        with open(output_path, "w") as f:
            json.dump(self.data, f, indent=2)
        print(f"Dataset saved to {output_path}")


class Preprocessor:
    """Xử lí dataset annotated sang COCO forrmat"""

    def __init__(self, annotated_dir, processed_dir):
        self.annotated_dir = annotated_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        self.dataset = COCODataset()

    def process_mask(self, mask_file):
        base_name = mask_file.replace('_mask.png', '')
        img_file = base_name + '.jpg'
        img_path = os.path.join(self.annotated_dir, img_file)
        mask_path = os.path.join(self.annotated_dir, mask_file)
        if not os.path.exists(img_path):
            return

        # Lấy thông tin ảnh
        img_id = len(self.dataset.data["images"]) + 1
        image = Image.open(img_path)
        width, height = image.size
        self.dataset.add_image(img_id, img_file, width, height)

        # Xử lí mask
        mask = cv2.imread(mask_path, 0)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            ann_id = len(self.dataset.data["annotations"]) + 1
            annotation = Annotation(contour, img_id, ann_id)
            self.dataset.add_annotation(annotation)

    def process_dataset(self):
        for mask_file in os.listdir(self.annotated_dir):
            if mask_file.endswith('_mask.png'):
                self.process_mask(mask_file)
        output_path = os.path.join(self.processed_dir, "dataset.json")
        self.dataset.save(output_path)


if __name__ == "__main__":
    preprocessor = Preprocessor("data/annotated", "data/processed")
    preprocessor.process_dataset()
