# Dự án Nhận diện Phương tiện & Biển số xe bằng YOLO26 (YouTube Live Stream)

Dự án này là hệ thống Computer Vision hoàn chỉnh cho phép:
1. **Kết nối thời gian thực** tới một luồng YouTube Live Stream (ví dụ: Camera giao thông) hoặc video YouTube thông thường để tự động chụp ảnh trích xuất làm tập dữ liệu (Dataset).
2. **Tự động dán nhãn (Auto-labeling)** bằng mô hình pretrained YOLO26 để gán nhãn cho các phương tiện: Ô tô (car), Xe máy (motorcycle), Xe tải (truck), Xe buýt (bus).
3. **Huấn luyện (Training)** mô hình YOLO26 trên tập dữ liệu tùy chỉnh đó (tự động phát hiện và hỗ trợ cả huấn luyện trên CPU hoặc GPU CUDA).
4. **Suy luận kiểm thử (Inference)** vẽ các hộp nhận diện (bounding box) trực quan có mã màu riêng biệt cho từng lớp đối tượng.

---

## 🚀 1. Cấu trúc Dự án

```text
computer_vision/
│
├── config/
│   └── dataset.yaml           # Cấu hình đường dẫn dataset và danh sách lớp đối tượng cho YOLO
│
├── src/
│   ├── data_downloader.py     # Phân tích & Tải/Trích xuất khung ảnh từ YouTube Live/VOD
│   ├── auto_label.py          # Tự động gán nhãn các lớp xe cộ bằng YOLO26 pretrained
│   ├── train.py               # Huấn luyện mô hình YOLO26 (hỗ trợ CPU/CUDA GPU)
│   └── inference.py           # Chạy suy luận nhận diện phương tiện trên ảnh/video
│
├── dataset/                   # Thư mục chứa hình ảnh và file nhãn đã gán
│   ├── images/
│   │   ├── train/             # Ảnh dùng để Train
│   │   └── val/               # Ảnh dùng để Validation
│   └── labels/
│       ├── train/             # Nhãn (.txt) tương ứng của ảnh Train
│       └── val/               # Nhãn (.txt) tương ứng của ảnh Val
│
├── runs/                      # Kết quả huấn luyện và suy luận
│   ├── detect/
│   │   ├── yolo26_traffic/    # Chứa kết quả huấn luyện (file best.pt, đồ thị loss...)
│   │   └── inference_results/ # Kết quả vẽ bounding box kiểm thử
│
├── requirements.txt           # Các thư viện Python cần thiết
└── README.md                  # Tài liệu hướng dẫn sử dụng (File này)
```

---

## 🛠️ 2. Cài đặt Môi trường

Đảm bảo bạn đã cài đặt Python (phiên bản khuyến nghị: >=3.10). 
Mở Terminal / PowerShell tại thư mục dự án và chạy lệnh sau để cài đặt các thư viện:

```bash
pip install -r requirements.txt
```

---

## 📸 3. Cách Vận hành Dự án

### Bước 1: Trích xuất Dữ liệu ảnh từ YouTube Live Stream
Mở file `src/data_downloader.py` và cập nhật đường dẫn video/live stream tại biến `youtube_url` trong khối `if __name__ == "__main__":`. Sau đó chạy lệnh:

```bash
python src/data_downloader.py
```
*   **Nếu là Live Stream:** Chương trình sẽ tự động lấy link luồng HLS trực tiếp (`.m3u8`) và chụp 1 ảnh mỗi 2 giây trong vòng 5 phút (300 giây). Bạn có thể bấm phím `'q'` tại cửa sổ hiển thị để dừng chụp sớm bất kỳ lúc nào.
*   **Nếu là Video thường:** Chương trình tải video về máy và trích xuất ảnh.

### Bước 2: Tự động dán nhãn Phương tiện
Để tiết kiệm thời gian vẽ tay cho các phương tiện phổ biến, hãy chạy script dán nhãn tự động bằng mô hình YOLO26 pretrained trên COCO:

```bash
python src/auto_label.py
```
Script sẽ tự quét qua toàn bộ ảnh trong `dataset/images/` và sinh ra các file `.txt` tương ứng trong `dataset/labels/` chứa tọa độ chuẩn hóa của: `car` (0), `motorcycle` (1), `truck` (2), và `bus` (3).

### Bước 3: Gán nhãn Biển số xe (License Plate - Lớp số 4)
Vì mô hình COCO gốc không hỗ trợ phát hiện biển số xe, bạn cần thực hiện gán nhãn cho lớp `license_plate` (class `4`):
1. Cài đặt công cụ gán nhãn như **labelImg** (`pip install labelImg` rồi gõ lệnh `labelImg` để mở giao diện) hoặc sử dụng các nền tảng online như **Roboflow**, **CVAT**.
2. Mở thư mục `dataset/images/` và load danh sách nhãn đã có sẵn từ `dataset/labels/`.
3. Vẽ thêm bounding box bao quanh các biển số xe hiển thị trên hình ảnh và gán nhãn là `license_plate` (ID lớp là `4`).
4. Lưu đè lại file `.txt`.

### Bước 4: Huấn luyện Mô hình YOLO26 Custom
Sau khi đã chuẩn bị xong ảnh và nhãn, tiến hành huấn luyện bằng lệnh:

```bash
python src/train.py
```
*   Script sẽ tự động phát hiện nếu máy tính có card đồ họa NVIDIA (CUDA) để huấn luyện trên GPU nhằm tăng tốc độ, nếu không sẽ tự động chạy trên CPU.
*   Bạn có thể thay đổi số lượng `epochs` và kích thước `batch` trong file `src/train.py` để phù hợp với tài nguyên máy của mình.
*   Sau khi kết thúc, file trọng số tốt nhất sẽ được lưu tại: `runs/detect/yolo26_traffic/vehicle_detector/weights/best.pt`.

### Bước 5: Chạy thử Nghiệm Suy luận (Inference)
Kiểm thử mô hình của bạn trên tập validation hoặc các hình ảnh mới:

```bash
python src/inference.py
```
Kết quả ảnh dự đoán đã được khoanh vùng và dán nhãn màu sắc bắt mắt sẽ được xuất ra thư mục: `runs/detect/inference_results/`.

---

## 🎨 Phân loại mã màu Nhận diện
*   🟢 **car (ô tô):** Màu xanh lá (Green)
*   🔵 **motorcycle (xe máy):** Màu xanh dương (Blue)
*   🟡 **truck (xe tải):** Màu vàng (Yellow)
*   🟠 **bus (xe buýt):** Màu cam (Orange)
*   🔴 **license_plate (biển số xe):** Màu đỏ (Red)
