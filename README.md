MÔ TẢ PROJECT_DETECT_OBJECT 

NHỮNG LƯU Ý TRƯỚC KHI CODE 

* LƯU Ý KHI CODE LÀ SỬ DỤNG OOP NHA, NÓI CHUNG CỨ SOURCE NÀO DÙNG OPP ĐƯỢC THÌ DÙNG HẾT:

    - TẠO MỘT MÔI TRƯỜNG ẢO TRƯỚC. 
    - NHỮNG THƯ VIỆN CẦN TÀI THÌ TRONG requirement.txt 

    - KHUYẾN NGHỊ NÊN SỬ DỤNG PYTHON 3.10.11  

* File PROJECT_DETECT_OBJECT/scripts/annote_data.py dùng để chuyền dữ liệu dạng thô ở /PROJECT_DETECT_OBJECT/data/raw thành dữ liệu để sử dụng cho /PROJECT_DETECT_OBJECT/data/annotated ---- VÌ data/raw CHỨA RẤT NHIÊU DỮ LIỆU HỖN TẠP NÊN BẮT CHUỘC FILE annote_data phải code để xí lí được nhiều file.

* File /PROJECT_DETECT_OBJECT/data/processed dùng để lưu trữ dữ liệu dạng json (dùng để train model - chuyền từ data/annotated qua json).

    Ex: PROJECT_DETECT_OBJECT/scripts/annote_data.py (FILE CODE VỪA CÓ THỂ SỬ LÝ DỮ LIỆU HÌNH ẢNH (NHIỀU ĐỊNH DẠNG) VÀ VIDEO )
                import os
                import numpy as np
                from PIL import Image
                import cv2
                from sam2.build_sam import build_sam2
                from sam2.sam2_video_predictor import SAM2VideoPredictor
                from scripts.automatic_mask_generator import SamAutomaticMaskGenerator

                class SAM2Model:
                    def __init__(self, checkpoint_path, config_path):
                        self.model = build_sam2(config_path, checkpoint_path)
                        self.mask_generator = SamAutomaticMaskGenerator(self.model)
                        self.video_predictor = SAM2VideoPredictor(self.model)  # Cho video

                    def generate_masks(self, image):
                        return self.mask_generator.generate(image)

                    def predict_video_frame(self, frame, prev_mask=None):
                        """Dự đoán mask cho frame video."""
                        return self.video_predictor.predict(frame, prev_mask)

                class DataProcessor:
                    def __init__(self, raw_dir, annotated_dir):
                        self.raw_dir = raw_dir
                        self.annotated_dir = annotated_dir
                        os.makedirs(self.annotated_dir, exist_ok=True)

                    def process_image(self, image_path, category, filename):
                        image = np.array(Image.open(image_path).convert("RGB"))
                        masks = self.model.generate_masks(image)
                        if masks:
                            best_mask = max(masks, key=lambda x: x['score'])['segmentation']
                            mask_filename = f"{category}_{filename[:-4]}_mask.png"
                            mask_path = os.path.join(self.annotated_dir, mask_filename)
                            cv2.imwrite(mask_path, best_mask.astype(np.uint8) * 255)
                            os.system(f"cp {image_path} {self.annotated_dir}")

                    def process_video(self, video_path, category, filename):
                        """Xử lý video và lưu masks cho từng frame."""
                        cap = cv2.VideoCapture(video_path)
                        prev_mask = None
                        frame_count = 0

                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            mask = self.model.predict_video_frame(frame, prev_mask)
                            if mask is not None:
                                prev_mask = mask
                                mask_filename = f"{category}_{filename[:-4]}_frame_{frame_count}_mask.png"
                                mask_path = os.path.join(self.annotated_dir, mask_filename)
                                cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)
                            frame_count += 1
                        cap.release()

                    def process_dataset(self, model):
                        self.model = model
                        for category in os.listdir(self.raw_dir):
                            category_path = os.path.join(self.raw_dir, category)
                            if os.path.isdir(category_path):
                                for file_name in os.listdir(category_path):
                                    file_path = os.path.join(category_path, file_name)
                                    if file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                                        self.process_image(file_path, category, file_name)
                                    elif file_name.endswith(('.mp4', '.avi')):  # Thêm định dạng video
                                        self.process_video(file_path, category, file_name)

                class AnnotationPipeline:
                    def __init__(self, checkpoint_path, config_path, raw_dir, annotated_dir):
                        self.model = SAM2Model(checkpoint_path, config_path)
                        self.processor = DataProcessor(raw_dir, annotated_dir)

                    def run(self):
                        self.processor.process_dataset(self.model.model)
                        print("Annotation completed! Check data/annotated/ for results.")

                if __name__ == "__main__":
                    pipeline = AnnotationPipeline(
                        checkpoint_path="checkpoints/sam2.1_hiera_large.pt",
                        config_path="configs/sam2.1_hiera_l.yaml",
                        raw_dir="data/raw",
                        annotated_dir="data/annotated"
                    )
                    pipeline.run()
    Ex: PROJECT_DETECT_OBJECT/scripts/preprocess_data.py
                import os
                import json
                import numpy as np
                import cv2
                from PIL import Image

                class Preprocessor:
                    def __init__(self, annotated_dir, processed_dir):
                        self.annotated_dir = annotated_dir
                        self.processed_dir = processed_dir
                        os.makedirs(self.processed_dir, exist_ok=True)

                    def process_mask(self, mask_file):
                        base_name = mask_file.replace('_mask.png', '')
                        category = base_name.split('_')[0]
                        img_file = base_name + '.jpg'  # Chinh if khac
                        img_path = os.path.join(self.annotated_dir, img_file)
                        mask_path = os.path.join(self.annotated_dir, mask_file)

                        if os.path.exists(img_path):
                            img_id = len(self.coco_data["images"]) + 1
                            image = Image.open(img_path)
                            width, height = image.size

                            self.coco_data["images"].append({"id": img_id, "file_name": img_file, "width": width, "height": height})
                            mask = cv2.imread(mask_path, 0)
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                for contour in contours:
                                    contour = contour.squeeze().tolist()
                                    ann_id = len(self.coco_data["annotations"]) + 1
                                    self.coco_data["annotations"].append({
                                        "id": ann_id, "image_id": img_id, "category_id": 1,
                                        "segmentation": [contour], "area": cv2.contourArea(contour),
                                        "bbox": [int(min(x) for x in contour), int(min(y for x, y in contour)),
                                                int(max(x) for x in contour) - int(min(x) for x in contour),
                                                int(max(y for x, y in contour) - min(y for x, y in contour))],
                                        "iscrowd": 0
                                    })

                    def process_dataset(self):
                        self.coco_data = {
                            "info": {"description": "Custom SAM2 Dataset", "date_created": "2025-10-01"},
                            "licenses": [], "images": [], "annotations": [], "categories": [{"id": 1, "name": "object"}]
                        }
                        for mask_file in os.listdir(self.annotated_dir):
                            if mask_file.endswith('_mask.png'):
                                self.process_mask(mask_file)

                        output_path = os.path.join(self.processed_dir, "dataset.json")
                        with open(output_path, "w") as f:
                            json.dump(self.coco_data, f)
                        print(f"Preprocessing completed! Check {output_path} for the result.")

                if __name__ == "__main__":
                    preprocessor = Preprocessor("data/annotated", "data/processed")
                    preprocessor.process_dataset()


* File PROJECT_DETECT_OBJECT/scripts/train.py dùng để train model.
    Ex:
        import torch
        from sam2.build_sam import build_sam2
        import os

        class SAM2Trainer:
            def __init__(self, config_path, checkpoint_path, processed_data):
                self.model = build_sam2(config_path, checkpoint_path)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                self.processed_data = processed_data  # Đường dẫn dataset.json
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

            def load_data(self):
                # Giả sử đọc JSON, cần code thêm để load dataset
                pass  # Thêm code load từ data/processed/

            def train_epoch(self):
                self.model.train()
                # Code training loop
                pass  # Thêm code forward, loss, backward

            def train(self, epochs=10):
                for epoch in range(epochs):
                    self.train_epoch()
                    print(f"Epoch {epoch+1}/{epochs} completed")
                torch.save(self.model.state_dict(), "checkpoints/finetuned_model.pt")

        if __name__ == "__main__":
            trainer = SAM2Trainer(
                config_path="configs/sam2.1_hiera_l.yaml",
                checkpoint_path="checkpoints/sam2.1_hiera_large.pt",
                processed_data="data/processed/dataset.json"
            )
            trainer.train()

* FILE PROJECT_DETECT_OBJECT/scripts/train.py dùng để  WEBCAM
    Ex:
        import cv2
        import torch
        from sam2.build_sam import build_sam2

        class WebcamInferencer:
            def __init__(self, config_path, checkpoint_path):
                self.model = build_sam2(config_path, checkpoint_path)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                self.cap = cv2.VideoCapture(0)

            def infer_frame(self, frame):
                # Giả sử generate mask (cần chỉnh với predictor)
                mask = self.model.predict(frame)  # Cần implement
                return mask

            def run(self):
                while self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    mask = self.infer_frame(frame)
                    if mask is not None:
                        frame[mask > 0] = [0, 255, 0]  # Highlight mask
                    cv2.imshow("Webcam Inference", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                self.cap.release()
                cv2.destroyAllWindows()

        if __name__ == "__main__":
            inferencer = WebcamInferencer(
                config_path="configs/sam2.1_hiera_l.yaml",
                checkpoint_path="checkpoints/finetuned_model.pt"  # Sử dụng model fine-tuned
            )
            inferencer.run()