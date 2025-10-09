# Dữ Liệu Đã Phân Loại Theo Cấu Trúc Phân Cấp - SAM2

## ✅ Hoàn Thành!

Dữ liệu đã được tổ chức theo cấu trúc phân cấp chuẩn SAM2:
```
category/object_type/object_typeXXX.ext
```

## 📊 Thống Kê Dataset

- **Tổng số ảnh**: 1,017 images
- **Số categories**: 12 categories  
- **Số object types**: 465 loại đồ vật khác nhau
- **Cấu trúc**: 3 cấp độ (category → object_type → files)

## 📁 Cấu Trúc Thư Mục

### Ví dụ cấu trúc:

```
dataset/
├── appliances/
│   ├── airconditioner/
│   │   ├── airconditioner001.jpeg
│   │   ├── airconditioner002.jpg
│   │   └── airconditioner003.jpeg
│   ├── refrigerator/
│   │   └── refrigerator001.jpg
│   └── washingmachine/
│       ├── washingmachine001.jpg
│       └── washingmachine002.jpg
├── furniture/
│   ├── chair/
│   │   ├── chair001.jpeg
│   │   └── chair002.jpg
│   └── table/
│       └── table001.jpg
└── ...
```

## 📈 Chi Tiết Categories

### 1. **Appliances** (133 images - 31 object types)
Thiết bị điện, gia dụng
- airconditioner (5), refrigerator (1), fridge (45)
- washingmachine (5), iron (9), fan (4)
- electricstove (9), ricecooker (1), microwave (1)
- vacuum_cleaner (1), và nhiều hơn...

### 2. **Bathroom Items** (18 images - 11 object types)
Đồ dùng nhà tắm
- toilet (4), bathtub (1), bucket (1)
- toothbrush (3), toothpaste (3), hanger (1)
- toilet_paper (1), và nhiều hơn...

### 3. **Bedding** (6 images - 3 object types)
Đồ giường, nội thất phòng ngủ
- bed (1), blanket (2), mirror (3)

### 4. **Cleaning Supplies** (19 images - 7 object types)
Đồ vệ sinh, làm sạch
- broom (5), mop (4), carpet (2)
- fabric_softener (5), basket (1), dustpan (1)

### 5. **Decor** (9 images - 4 object types)
Đồ trang trí
- lamp (5), paper_fan (2)
- alarmclock (1), monitorarm (1)

### 6. **Electronics** (64 images - 28 object types)
Thiết bị điện tử
- laptop (6), remote (9), television (4)
- headphone (3), keyboard (3), printer (3)
- tablet (2), monitor (1), và nhiều hơn...

### 7. **Furniture** (218 images - 214 object types)
Nội thất - Đa dạng nhất
- chair (4), table (1), wardrobe (4)
- window (1), shoerack (1), armchair (1)
- Rất nhiều furniture items khác...

### 8. **Kitchenware** (108 images - 47 object types)
Đồ dùng nhà bếp
- bowl (3), cup (5), plate (3)
- spoon (4), fork (3), knife (2)
- pan (2), pot (4), wok (2)
- chopsticks (2), và nhiều hơn...

### 9. **Learning Tools** (90 images - 42 object types)
Dụng cụ học tập
- pen (5), pencil (5), notebook (3)
- ruler (4), scissors (4), eraser (3)
- book (2), sharpener (2), và nhiều hơn...

### 10. **Outdoor & Utility** (6 images - 6 object types)
Đồ dùng ngoài trời
- bicycle (1), car (1), ladder (1)
- wateringcan (1), antenna (1), airconditioner (1)

### 11. **Personal Belongings** (109 images - 76 object types)
Đồ dùng cá nhân
- bag (5), backpack (3), wallet (7)
- shoes (5), sandals (2), boots (1)
- watch (3), glasses (2), belt (1)
- clothing items: jacket (2), jeans (1), sweater (1)
- và nhiều hơn...

### 12. **Synthetic** (237 images - 34 object types)
Dữ liệu synthetic từ nhiều categories
- cooking_utensil (12), kitchen_tool (12)
- electronic_device (8), tech_item (8)
- furniture_piece (6), decorative_vase (7)
- và nhiều loại khác...

## 🔧 Quy Tắc Đặt Tên

### Object Type (Thư mục)
- Tên được chuẩn hóa từ tên file gốc
- Chuyển tiếng Việt sang tiếng Anh (một phần)
- Loại bỏ ký tự đặc biệt
- Chuyển thành chữ thường

### File Name (Tên file)
Format: `{object_type}{counter}.{ext}`

Ví dụ:
- `airconditioner001.jpeg`
- `refrigerator001.jpg`  
- `chair015.jpg`

Counter: 3 chữ số (001, 002, 003,...)

## 📋 Files Metadata

### `dataset_info.json`
Chứa thông tin chi tiết:
- Dataset name và structure
- Total images: 1,017
- Categories với object types breakdown
- File mappings (original → new path)

### `categories.txt`
Danh sách 12 categories:
```
appliances
bathroom_items
bedding
cleaning_supplies
decor
electronics
furniture
kitchenware
learning_tools
outdoor_utility
personal_belongings
synthetic
```

### `object_types.txt`
Danh sách đầy đủ 465 object types

## 🚀 Sử Dụng với SAM2

### Bước 1: Generate Annotations
```bash
python scripts/preprocess_data.py \
  --checkpoint checkpoints/sam2.1_hiera_large.pt \
  --config configs/sam2.1_hiera_l.yaml \
  --raw_dir data/processed \
  --annotated_dir data/annotated \
  --processed_dir data/final \
  --device cuda
```

### Bước 2: Training SAM2
```bash
python scripts/train.py \
  --config configs/sam2.1_hiera_l.yaml \
  --data_dir data/final \
  --output_dir outputs/
```

## 📝 Lưu Ý

### ✅ Đã Hoàn Thành:
- Phân loại 12 categories
- Tổ chức theo cấu trúc phân cấp 3 cấp
- Chuẩn hóa tên file
- Chuyển đổi một phần tên tiếng Việt sang tiếng Anh
- Tạo metadata files

### ⚠️ Cần Lưu Ý:
- Một số tên tiếng Việt vẫn còn (ghế_gỗ, giày, dép, v.v.)
- Một số file có tên là UUID/hash (do tên gốc không có ý nghĩa)
- Furniture category có rất nhiều object types (214 loại)

### 🔄 Dữ Liệu Gốc:
- Dữ liệu gốc: `data/raw/`
- Dữ liệu đã tổ chức: `data/processed/`
- File mapping giúp truy vết từ gốc đến processed

## 📁 Tổng Kết Cấu Trúc

```
data/
├── raw/                    # Dữ liệu gốc (giữ nguyên)
├── organized/              # Dữ liệu chuẩn hóa đơn giản
└── processed/              # Dữ liệu phân cấp chi tiết ⭐
    ├── {category}/
    │   └── {object_type}/
    │       └── {object_type}{counter}.{ext}
    ├── dataset_info.json
    ├── categories.txt
    ├── object_types.txt
    └── README.md (this file)
```

## 🎯 Kết Luận

Dữ liệu đã được tổ chức thành công theo cấu trúc phân cấp chuẩn SAM2:
- ✅ 1,017 images được phân loại
- ✅ 12 categories chính
- ✅ 465 object types chi tiết
- ✅ Cấu trúc 3 cấp độ rõ ràng
- ✅ Metadata đầy đủ

**Dataset sẵn sàng cho bước tiếp theo: SAM2 Annotation & Training!**

---

*Generated by: reorganize_data_hierarchical.py*  
*Date: 2024*  
*Project: PROJECT_DETECT_OBJECT*

