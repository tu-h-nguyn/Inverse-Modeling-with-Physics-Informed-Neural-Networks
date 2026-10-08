# Rà soát IP_PINNs.zip

## Những gì đã kiểm tra trực tiếp

- ZIP gồm một notebook, `runs_raw.csv`, `summary.csv` và một bảng LaTeX. Notebook gốc chưa lưu execution count hoặc output.
- Có 36 dòng CSV cũ, gom thành 12 nhóm, mỗi nhóm ba seed. Tính lại toàn bộ trung bình và độ lệch chuẩn mẫu khớp `summary.csv`.
- PDE và điều kiện biên/đầu nhất quán. Kiểm tra đạo hàm trong notebook cho residual tối đa khoảng `9.5e-7` với α đúng, và khoảng `1.8` với α sai.
- Giữ nguyên phép đo, thay `alpha_true` và thay toàn bộ `u_clean`: tổng loss không đổi. Chỉ thành phần PDE truyền gradient trực tiếp tới `log_alpha`.
- Lặp lại cấu hình 100 bước với RNG toàn cục bị thay đổi vẫn thu được cùng α. Worker song song cũng tái tạo đúng α này.

## Những điểm cần lưu ý trong mã gốc

1. **`Config(n_obs=1)` được cho phép nhưng tạo NaN** (cell 4 và 8). `u_clean.std()` dùng độ lệch chuẩn mẫu, không xác định với một phần tử. Đã tái hiện lỗi `obs.u contains NaN/Inf`. Với notebook hiện tại, dùng ít nhất hai phép đo; các thí nghiệm mặc định dùng 20, 50 hoặc 100.
2. **Một số số lượng bằng 0 chưa bị chặn trong `Config`** (cell 4 và 8). `n_colloc=0` hoặc `n_bc=0` đi đến lỗi reduction trên tensor rỗng. Dùng số điểm dương, `n_bc` chẵn; số bước, `log_every`, width và depth phải dương. Nên thay assert bằng kiểm tra `ValueError` đầy đủ nếu phát triển thành thư viện dùng chung.
3. **Nhiễu giữa các tập quan sát lồng nhau không hoàn toàn giống nhau** (cell 8). Tọa độ và mẫu nhiễu chuẩn hóa dùng chung, nhưng độ lệch chuẩn nhiệt độ được tính lại theo kích thước tập. Vì vậy 20 phép đo noisy không nhất thiết bằng 20 hàng đầu của tập 50 phép đo. Đây là thay đổi SNR theo quy ước của notebook, không phải bằng chứng lỗi RNG; cần nói rõ khi so sánh số lượng sensor.
4. **L-BFGS có thể dừng sớm trong float32** (cell 18 và 37). Trạng thái `ok` chỉ cho biết loss cuối hữu hạn và không tăng; không chứng minh hội tụ hoặc α chính xác. Notebook có cơ chế quay lại trọng số Adam nếu bước L-BFGS không đạt kiểm tra. Chẩn đoán nguyên nhân chính xác của dừng sớm cần thí nghiệm riêng, không được coi giả thuyết về độ chính xác số là kết luận đã chứng minh.
5. **Khả năng định danh phụ thuộc vị trí và thời gian đo**. Đo chỉ ở t=0 hoặc biên không cung cấp thông tin về α. Dữ liệu tổng hợp hiện tại dùng bố trí ngẫu nhiên trong miền; chưa kiểm chứng với dữ liệu thực, sai lệch mô hình, α biến thiên theo không gian hoặc điều kiện đầu chưa biết.

Mã khoa học gốc được giữ nguyên để so sánh có nguồn gốc rõ ràng. Những điểm trên là giới hạn đã rà soát, không phải các sửa lỗi đã được áp dụng. Các script môi trường chỉ bổ sung cách thực thi và lưu đầu ra.

## Tính toàn vẹn

- SHA256 ZIP: `5b8cfb16b6a49247787fc8d56ba8c5e7b1f11577699d97377f6ce48ca307a325`.
- SHA256 notebook gốc: `564cf427094f47eb507d18d77084b80a846f04127e5eb67752be9ee5a44229ff`.
- Các bảng cũ được giữ ở `reference_results/`; kết quả chạy mới nằm trong thư mục riêng `runs/verified-20261008/`.

Số liệu và kiểm tra đầu ra của lần chạy mới được ghi trong `VALIDATION.md` khi chạy hoàn tất.
