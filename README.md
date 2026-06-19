# 🚦 Hệ Thống Giám Sát Giao Thông & Nhận Diện Biển Số Xe (YOLO26 + VietOCR)

Dự án này là một hệ thống Computer Vision hoàn chỉnh, kết hợp phát hiện phương tiện, theo dõi hành trình (tracking), định vị biển số và nhận diện ký tự quang học (OCR) thời gian thực từ các nguồn phát trực tiếp YouTube Live Stream (camera giao thông đô thị).

## 🌟 Tính năng nổi bật

1. **Dashboard Giám Sát Trực Tuyến**: Giao diện Flask Web Server cho phép xem đồng thời 6 luồng camera giao thông trực tiếp với tốc độ khung hình cao.
2. **Nhận Diện Phương Tiện & Theo Dõi**: Sử dụng YOLO26 kết hợp thuật toán **ByteTrack** để bám đuôi đối tượng, đồng thời làm mượt hộp giới hạn (Bounding Box Smoothing) và bình chọn lớp (Class Smoothing) qua nhiều khung hình.
3. **Phát Hiện Biển Số Xe Hai Giai Đoạn (Cascaded Detection)**: Nhận diện biển số xe ngay bên trong vùng cắt của phương tiện đã được phát hiện để tối ưu độ chính xác.
4. **Nhận Diện Chữ Số Biển Số Xe (OCR)**:
   * **Deskew**: Tự động căn chỉnh góc nghiêng của biển số bằng bộ lọc Hough Lines.
   * **OCR Backend**: Linh hoạt chuyển đổi giữa **PaddleOCR** (ưu tiên) và **EasyOCR**.
   * **Phân biệt biển vàng**: Kiểm tra không gian màu HSV để tự động gắn nhãn "Biển vàng" (xe kinh doanh vận tải) hoặc "Biển số" thường.
   * **Chống nhấp nháy (Anti-flickering)**: Lưu trữ bộ nhớ cache theo `track_id` để biển số hiển thị ổn định, giảm tải cho CPU.

---

## 📂 Cấu Trúc Dự Án

* [src/web_server.py](file:///c:/dev/computer_vision/src/web_server.py): Flask Web Server chính quản lý luồng livestream, xử lý hình ảnh và thống kê thời gian thực.
* [src/train.py](file:///c:/dev/computer_vision/src/train.py): Script huấn luyện mô hình YOLO26 phát hiện phương tiện.
* [src/train_plate.py](file:///c:/dev/computer_vision/src/train_plate.py): Script huấn luyện mô hình YOLO26 phát hiện biển số xe.
* [src/data_downloader.py](file:///c:/dev/computer_vision/src/data_downloader.py): Trích xuất ảnh mẫu từ các luồng YouTube Livestream.
* [src/auto_label.py](file:///c:/dev/computer_vision/src/auto_label.py): Tự động gán nhãn phương tiện bằng mô hình YOLO26 pretrained.

---

## 🛠️ Cài Đặt Môi Trường

Kích hoạt môi trường ảo `.venv` và cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### Kích hoạt GPU (CUDA) để huấn luyện & chạy thời gian thực:
Nếu máy tính của bạn sử dụng card đồ họa rời NVIDIA (ví dụ: RTX 4060) và muốn tận dụng GPU, hãy cài đặt phiên bản PyTorch hỗ trợ CUDA bằng lệnh:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

---

## 🚀 Hướng Dẫn Vận Hành

### 1. Khởi chạy Web Dashboard
Để chạy máy chủ web và xem luồng giám sát camera thời gian thực, hãy thực thi lệnh:

```bash
python src/web_server.py
```
Mở trình duyệt và truy cập: `http://localhost:5000`

### 2. Thu thập & Tự động gán nhãn dữ liệu (Nếu muốn mở rộng Dataset)
* **Chụp ảnh từ camera**: Chạy lệnh để chụp khung hình từ luồng stream:
  ```bash
  python src/data_downloader.py
  ```
* **Tự động dán nhãn**: Chạy lệnh để mô hình tự động khoanh vùng và gán nhãn trước các phương tiện:
  ```bash
  python src/auto_label.py
  ```

### 3. Huấn luyện mô hình tùy chỉnh
* **Huấn luyện mô hình phương tiện**:
  ```bash
  python src/train.py
  ```
* **Huấn luyện mô hình biển số**:
  ```bash
  python src/train_plate.py
  ```

---

## 🎨 Phân Loại Nhận Diện

* **Ô tô (car)**: Khung màu xanh lá 🟢
* **Xe máy (motorcycle)**: Khung màu xanh dương 🔵
* **Xe tải (truck)**: Khung màu vàng nhạt 🟡
* **Xe buýt (bus)**: Khung màu cam 🟠
* **Biển số thường**: Khung màu đỏ 🔴
* **Biển số vàng**: Khung màu vàng tươi 🟡 (Có ghi chú `Bien vang` trên hộp)
