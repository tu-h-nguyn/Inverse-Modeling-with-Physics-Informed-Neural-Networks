# Kết quả kiểm chứng ngày 08/10/2026

Đã chạy trọn notebook gốc trên CPU: 18/18 cell mã nguồn hoàn tất, cộng một cell điều phối song song và một cell lưu artifact. Không có output lỗi. Chạy đầy đủ `QUICK_MODE=False`, seed 0–2, 5.000 bước Adam; các cấu hình L-BFGS dùng tối đa 1.000 bước theo notebook.

Có **27 cấu hình huấn luyện khác nhau**, trình bày thành **36 dòng** trong 12 nhóm. Baseline trùng giữa các nhóm được tái sử dụng; 36 dòng không phải 36 mô hình độc lập.

## Baseline đã chạy và nạp lại

| Chỉ số (seed 0) | Giá trị |
| --- | --- |
| α thật | 0,5 |
| α khởi tạo | 0,1 |
| α suy ra | 0,4991582632 |
| Sai số tương đối α | 0,168347% |
| Sai số tương đối L2 của trường | 0,215240% |
| MSE trường | 2,4161 × 10⁻⁷ |
| MSE residual trên lưới đánh giá | 1,3124 × 10⁻⁵ |

Checkpoint đã nạp lại bằng `weights_only=True` trong một tiến trình mới và tái tạo đúng năm chỉ số đã lưu. Dự đoán kiểm tra tại x=0,5, t=0,2 cho u≈0,37368548. Trường NPZ được đối chiếu độc lập với chỉ số trong manifest.

## Tổng hợp ba seed

Số sau dấu ± là độ lệch chuẩn mẫu, không phải khoảng tin cậy. Hai cột sai số được đổi sang phần trăm; CSV gốc lưu dạng tỷ lệ.

| Nhóm | Cấu hình | Sai số α (%) | Sai số L2 trường (%) | Residual MSE |
| --- | --- | ---: | ---: | ---: |
| noise | 0% noise | 0.4107 ± 0.2285 | 0.4048 ± 0.2041 | 3.119e-05 |
| noise | 1% noise | 0.3597 ± 0.1657 | 0.4239 ± 0.1816 | 3.150e-05 |
| noise | 5% noise | 0.3229 ± 0.1896 | 0.7417 ± 0.2182 | 3.025e-05 |
| n_obs | N_obs = 20 | 1.4006 ± 1.7379 | 1.0648 ± 1.1814 | 3.341e-05 |
| n_obs | N_obs = 50 | 0.3597 ± 0.1657 | 0.4239 ± 0.1816 | 3.150e-05 |
| n_obs | N_obs = 100 | 0.2784 ± 0.1525 | 0.4100 ± 0.2251 | 2.695e-05 |
| weights | balanced (1,1,1,1) | 0.3597 ± 0.1657 | 0.4239 ± 0.1816 | 3.150e-05 |
| weights | data-heavy (1,1,1,10) | 0.1111 ± 0.1309 | 0.4361 ± 0.0367 | 4.991e-05 |
| weights | physics-heavy (10,1,1,1) | 10.3275 ± 2.7783 | 6.2379 ± 1.4763 | 3.015e-05 |
| weights | physics-light (0.1,1,1,1) | 0.1770 ± 0.0942 | 0.2794 ± 0.0542 | 1.323e-04 |
| optimizer | Adam | 0.3597 ± 0.1657 | 0.4239 ± 0.1816 | 3.150e-05 |
| optimizer | Adam + L-BFGS | 0.2129 ± 0.2210 | 0.2936 ± 0.1938 | 1.914e-05 |

## Diễn giải

- Baseline ba seed có α trung bình 0,4982015 ± 0,0008287; sai số tương đối trung bình 0,3597%.
- Trọng số data-heavy cho sai số α trung bình khoảng 0,1111%, nhưng residual lớn hơn baseline. Đây là đánh đổi giữa khớp dữ liệu và phương trình.
- Trọng số physics-heavy có sai số α trung bình khoảng 10,3275% sau cùng ngân sách lặp, dù residual vẫn nhỏ. Cấu hình này không đạt độ chính xác của baseline; không coi việc chương trình chạy xong là hội tụ tốt.
- Với 20 phép đo, một seed có sai số α 3,41%. Nhiều phép đo hơn hoặc vị trí đo có thông tin tốt hơn có thể giúp, nhưng ba seed chưa đủ để kết luận thống kê tổng quát.
- Cả ba lượt L-BFGS có trạng thái `ok`, lần lượt 18, 688 và 10 lần đánh giá hàm. Mức cải thiện rõ nhất ở seed 1; không có bằng chứng nó luôn cải thiện mọi seed.

## Kiểm tra đầu ra

- Tính lại trung bình và độ lệch chuẩn từ 36 dòng raw: khớp toàn bộ 12 dòng summary.
- Mọi chỉ số hữu hạn, mọi α dương; baseline đạt sai số α <1%, sai số L2 <1% và residual MSE <10⁻⁴.
- Có đủ 6 PNG, notebook đã thực thi, HTML, CSV, bảng LaTeX, checkpoint, dữ liệu quan sát, trường đánh giá và manifest phiên bản.
- Các chỉ số số học của cả 36 dòng tái tạo đúng bảng tham khảo được gửi trong ZIP. Runtime khác và không được dùng làm điều kiện so khớp.
- Notebook gốc giữ nguyên SHA256; dữ liệu tham khảo được giữ riêng. Các lỗi đầu vào đã tái hiện được ghi trong `REVIEW_VI.md`, chưa sửa trong mã khoa học gốc.
- Script setup chạy lại thành công; `pip check` không báo xung đột. Giới hạn một thread/worker đã được kiểm tra cả khi tiến trình cha đặt OMP/MKL bằng 4.

## Giới hạn và trạng thái cloud

Đây là dữ liệu tổng hợp cho một hệ số vô hướng của phương trình nhiệt 1D. Chưa kiểm chứng với phép đo thực, α(x), miền 2D/3D, GPU hoặc một môi trường cloud mới được khôi phục.

Lần kiểm chứng dùng năm worker, mỗi worker một thread, trên quota tổng cộng bốn CPU. Chạy song song làm runtime từng job chịu tranh chấp tài nguyên; dùng chế độ tuần tự để so sánh tốc độ optimizer. Thời gian baseline đo được thay đổi từ khoảng 130 đến 209 giây giữa hai lần chạy; lượt tuần tự ban đầu đã được ngắt sau baseline để chuyển sang sweep song song, không tính là một sweep hoàn tất.

Trong phiên onboarding ban đầu, `install_script` và `start_skill` đã được lưu vào draft cấu hình cloud; phiên kiểm chứng này không thực hiện Publish. Repository được khởi tạo từ ZIP do remote chưa có commit khi bắt đầu. Mã nguồn và bộ kết quả đã kiểm chứng được đưa vào Git cùng tài liệu này; các cache và lần chạy dở được loại khỏi Git.

Bằng chứng máy đọc được: `runs/verified-20261008/verification.json`, `run_manifest.json`, `source_audit.json` và `checkpoint_reload_verification.json`.
