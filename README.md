# 🚦 Hệ Thống Giám Sát Giao Thông & Nhận Diện Biển Số Xe (YOLO26 + Gemini VLM)

Hệ thống giám sát giao thông đô thị thông minh thời gian thực tích hợp công nghệ AI tiên tiến, hỗ trợ giám sát luồng phương tiện trực tiếp và tự động phân tích biển số xe với độ chính xác cao nhờ sự kết hợp giữa mô hình Object Detection và Vision-Language Model (VLM).

---

## 🌟 Các Chức Năng Chính (Core Features)

1. **Dashboard Giám Sát Đa Luồng Thời Gian Thực (Multi-stream Monitoring)**
   * Hỗ trợ giải mã và hiển thị đồng thời 6 luồng camera giao thông độ phân giải cao trực tiếp từ YouTube Live Stream.
   * Cập nhật giao diện mượt mà với độ trễ thấp tối đa.

2. **Nhận Diện & Theo Dõi Phương Tiện Thông Minh (Vehicle Tracking & Association)**
   * Tự động phát hiện và phân loại các nhóm phương tiện giao thông phổ biến: Ô tô (`O to`), Xe máy (`Xe may`), Xe tải (`Xe tai`), Xe buýt (`Xe buyt`).
   * Sử dụng thuật toán **ByteTrack** để gán ID duy nhất (`track_id`) cho từng phương tiện di chuyển trong vùng quan sát.

3. **Làm Mượt Chuyển Động & Phân Loại Ổn Định (Smoothing Algorithms)**
   * **Bounding Box Smoothing**: Áp dụng thuật toán Moving Average (Trung bình động) trên 5 khung hình gần nhất để làm mịn các hộp giới hạn, triệt tiêu hiện tượng rung lắc hộp nhận diện.
   * **Class Smoothing**: Áp dụng cơ chế Majority Voting (Bỏ phiếu số đông) trên 10 khung hình để ổn định lớp phương tiện khi góc nhìn thay đổi.

4. **Phát Hiện Biển Số Xe Hai Giai Đoạn (Cascaded Plate Detection)**
   * Hạn chế sai số và nhiễu nền bằng cách chỉ kích hoạt mô hình phát hiện biển số xe bên trong vùng cắt ảnh (crop) của các phương tiện được phát hiện có kích thước hợp lệ.

5. **Nhận Diện Ký Tự Biển Số Độ Chính Xác Cao (Gemini VLM OCR)**
   * Sử dụng mô hình Vision-Language Model **Gemini 2.5 Pro** thông qua API để đọc nội dung biển số từ ảnh cắt.
   * Khả năng hiểu ngữ cảnh giúp đọc chính xác biển số kể cả trong các điều kiện bất lợi như ảnh bị mờ do chuyển động, góc nghiêng lớn, hoặc ánh sáng yếu.

6. **Tự Động Cân Chỉnh Góc Nghiêng (Deskewing)**
   * Sử dụng bộ lọc Hough Lines để tự động đo góc lệch của biển số xe so với phương ngang và tiến hành xoay ảnh góc nghiêng trước khi gửi đến VLM nhằm tối ưu kết quả đọc ký tự.

7. **Phân Loại Màu Biển Số (HSV Color Classification)**
   * Tự động phân tích không gian màu HSV trên vùng biển số để phân loại giữa biển số trắng thông thường (`Bien so`) và biển số vàng kinh doanh vận tải (`Bien vang`).

8. **Chống Nhấp Nháy & Tối Ưu Hóa Tần Suất Gọi API (Anti-flickering Cache)**
   * Lưu trữ bộ đệm cache biển số đã nhận diện theo `track_id` phương tiện.
   * Giữ kết quả biển số ổn định không bị nhấp nháy qua từng frame và giới hạn số lượt gọi Cloud API để tối ưu hóa chi phí cũng như giảm băng thông mạng.

9. **API Thống Kê Số Liệu Thời Gian Thực (Real-time Stats API)**
   * Cung cấp endpoint API JSON cập nhật liên tục lưu lượng từng loại phương tiện đang hoạt động trên mỗi camera.

---

## 🛠️ Công Nghệ Sử Dụng (Technologies & Stack)

* **Ngôn ngữ**: Python 3.14+ (Tận dụng hiệu năng mới nhất của Python).
* **Mô hình học máy & AI (Deep Learning & GenAI)**:
  * **Ultralytics YOLO26**: Mô hình Object Detection thế hệ mới nhất cho tốc độ suy luận cực nhanh trên CPU và GPU.
  * **ByteTrack**: Thuật toán bám đuôi đối tượng đa mục tiêu thời gian thực.
  * **Google Generative AI (Gemini 2.5 Pro)**: Mô hình ngôn ngữ lớn đa phương thức đóng vai trò làm VLM OCR chính xác vượt trội.
  * **PyTorch (CUDA GPU)**: Khung học sâu làm nền tảng chạy suy luận YOLO.
* **Xử lý ảnh & Tính toán (Computer Vision & Math)**:
  * **OpenCV**: Thư viện xử lý hình ảnh và video chính (Deskew, warpAffine, imencode, colorspaces).
  * **NumPy**: Thư viện tính toán ma trận để phân tích phân phối góc nghiêng và không gian màu HSV.
  * **Pillow (PIL)**: Chuyển đổi và thao tác định dạng ảnh đầu vào cho mô hình VLM.
* **Môi trường & Web Dashboard**:
  * **Flask**: Framework web tối giản để quản lý luồng camera và cung cấp JSON API.
  * **yt-dlp**: Công cụ phân tích và giải mã trực tiếp URL livestream sang luồng HLS (`.m3u8`).
  * **Python Dotenv**: Tự động cấu hình và nạp API key cục bộ an toàn.
  * **Logging**: Module ghi nhật ký hệ thống tiêu chuẩn lưu trữ log chạy dịch vụ.
