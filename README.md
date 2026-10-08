# PINN cho bài toán ngược phương trình nhiệt 1D

Khôi phục hệ số khuếch tán nhiệt **α > 0** từ các phép đo nhiệt độ thưa, có nhiễu. Mã khoa học được nhập nguyên bản từ `IP_PINNs.zip`; notebook gốc không bị sửa. Các script bổ sung giúp chạy toàn bộ notebook, lưu mô hình và xuất kết quả có thể tái lập.

Kết quả đã kiểm chứng: [VALIDATION.md](VALIDATION.md). Notebook có output nằm ở [runs/verified-20261008/inverse_pinn_heat_1d.executed.ipynb](runs/verified-20261008/inverse_pinn_heat_1d.executed.ipynb); bản HTML ở cùng thư mục.

## Bài toán và phạm vi

\[
u_t = \alpha u_{xx},\quad (x,t)\in[0,1]^2,\qquad
u(0,t)=u(1,t)=0,\quad u(x,0)=\sin(\pi x).
\]

Đây là **benchmark với dữ liệu tổng hợp**, không phải dữ liệu thực nghiệm: α thật bằng 0,5; giá trị khởi tạo bằng 0,1. Các CSV trong ZIP là kết quả thí nghiệm cũ, không phải tập phép đo đầu vào. Nghiệm giải tích chỉ tạo dữ liệu và đánh giá; loss chỉ dùng PDE, điều kiện biên, điều kiện đầu và phép đo có nhiễu.

Mạng `tanh` có 4 lớp ẩn × 50 neuron, học đồng thời trường nhiệt độ và `log(α)`. Mặc định: 50 phép đo, nhiễu 1%, 2.000 điểm collocation, 200 điểm biên, 200 điểm ban đầu, 5.000 bước Adam. Phần thí nghiệm so sánh thêm Adam + L-BFGS, ba mức nhiễu, ba số lượng phép đo và bốn bộ trọng số loss, mỗi cấu hình có ba seed.

## Chạy từ đầu

Đã kiểm tra trên Linux x86_64, Python 3.12, PyTorch 2.14.1 CPU. Không cần API key hoặc dịch vụ ngoài; việc cài đặt dùng PyPI và kho wheel CPU chính thức của PyTorch.

```bash
cd /workspace/Inverse-Modeling-with-Physics-Informed-Neural-Networks
bash scripts/setup.sh
.venv/bin/python scripts/run_notebook.py \
  --source notebooks/inverse_pinn_heat_1d.ipynb \
  --output runs/my-experiment \
  --workers 4
```

Trong môi trường cloud đã chuẩn bị, dùng `/workspace/pinn-venv/bin/python` thay cho `.venv/bin/python`; không cần cài lại. `--output` phải là một thư mục mới để không ghi đè kết quả. `--workers 1` chạy tuần tự; tối đa 5 worker, mỗi worker một thread. Máy cloud này có quota tổng cộng 4 CPU, nên dùng 4 worker cho các lần sau. Một baseline 5.000 bước mất khoảng 209 giây trong lần đo đầu tiên; toàn bộ sweep tốn hơn nhiều và phụ thuộc tải máy.

Script giữ nguyên các cell nguồn. Khi dùng nhiều worker, script thêm một cell điều phối trước sweep để chạy các cấu hình độc lập bằng đúng hàm huấn luyện trong notebook, rồi đưa kết quả vào cache vốn có. Baseline trùng cấu hình được tái sử dụng. Vì các lần chạy dùng chung CPU, **runtime trong sweep song song không thích hợp để kết luận optimizer nào nhanh hơn**; dùng `--workers 1` để so thời gian.

Ở chế độ song song, dòng `Part 2 training finished` của notebook gốc chỉ đo vòng tổng hợp sau khi cache đã được tính, nên có thể bằng 0 phút. Thời gian đó không phải tổng thời gian huấn luyện; các trường `runtime_s` trong CSV là thời gian từng job, có cả ảnh hưởng tranh chấp CPU.

Cũng có thể mở `notebooks/inverse_pinn_heat_1d.ipynb` trên Jupyter hoặc Colab và chọn **Run all**. Notebook gốc dùng `QUICK_MODE=False`; `QUICK_MODE=True` chỉ giảm sweep xuống hai seed và 2.000 bước, không giảm baseline. Không trộn kết quả quick với kết quả full.

## Đầu ra

Trong thư mục được truyền qua `--output`:

| File | Nội dung |
| --- | --- |
| `inverse_pinn_heat_1d.executed.ipynb` | Notebook đã chạy, có output và hình |
| `report.html` | Bản đọc tĩnh của notebook |
| `results/runs_raw.csv` | 36 dòng kết quả; 27 cấu hình huấn luyện khác nhau |
| `results/summary.csv` | 12 nhóm, trung bình và độ lệch chuẩn mẫu trên ba seed |
| `results/results_table.tex` | Bảng LaTeX; dùng `booktabs` khi ghép vào bài viết |
| `baseline_checkpoint.pt` | Trọng số mạng, cấu hình, dtype và chỉ số baseline |
| `baseline_fields.npz` | Trường dự đoán, trường chuẩn, sai số và residual trên lưới 101 × 101 |
| `observations.csv` | 50 phép đo tổng hợp `(x,t,u)` của baseline |
| `run_manifest.json` | Phiên bản thư viện, cấu hình, lịch sử loss/α và thông tin lần chạy |
| `figures/` | Các hình PNG xuất từ output notebook |
| `sweep_cache.json` | Các kết quả hoàn tất khi chạy song song |

Sau khi chạy notebook và có các định nghĩa `Config`, `InversePINN`, có thể nạp mô hình để dự đoán:

```python
saved = torch.load("baseline_checkpoint.pt", map_location="cpu", weights_only=True)
restored = InversePINN(Config(**saved["config"])).to(device="cpu", dtype=torch.float32)
restored.load_state_dict(saved["state_dict"])
restored.eval()
with torch.no_grad():
    u_hat = restored(torch.tensor([[0.5]]), torch.tensor([[0.2]])).item()
print(restored.alpha.item(), u_hat)
```

Ví dụ này dành cho checkpoint float32 của lần chạy đã xác minh; nạp trong thư mục chứa checkpoint hoặc truyền đường dẫn đầy đủ. Nó trả nhiệt độ tại x=0,5 và t=0,2, không chạy lại optimizer.

Nếu chạy lỗi hoặc bị ngắt, `partial.ipynb` giữ các output đã nhận; file này không chứng minh toàn bộ thí nghiệm đã hoàn tất. Một lần chạy thành công in `SUCCESS` và tạo `run_manifest.json` cùng notebook đã thực thi.

## Đọc kết quả đúng cách

- `reference_results/` giữ nguyên bảng từ ZIP. Phần Discussion trong notebook cũng mô tả lần chạy tham khảo cũ. Các kết luận cho lần chạy mới phải dựa trên `runs/.../results/` và manifest mới.
- Sai số tương đối của α, sai số trường nhiệt độ và residual PDE đo ba khía cạnh khác nhau. Residual nhỏ chưa đủ chứng minh α đúng.
- Các nhóm lặp baseline dùng chung kết quả, không phải các quan sát thống kê độc lập. Ba seed chỉ cho một ước lượng độ biến thiên sơ bộ.
- CRLB là giới hạn cho độ lệch chuẩn của một lớp estimator dưới giả thiết cụ thể; không phải ngưỡng dưới cho sai số tuyệt đối của từng lần chạy.
- Đây là thí nghiệm ngân sách lặp cố định, không đảm bảo mọi cấu hình đã hội tụ. Xem `REVIEW_VI.md` để biết các điểm cần cải thiện trước khi dùng dữ liệu hoặc cấu hình khác.

Môi trường cloud đã cô lập sẵn: dùng checkout hiện có, không tạo Git worktree. Repository lưu mã nguồn, tài liệu và bộ kết quả đã kiểm chứng trong `runs/verified-20261008/`. Các lần chạy mới, môi trường Python và cache được bỏ qua bởi `.gitignore`.
