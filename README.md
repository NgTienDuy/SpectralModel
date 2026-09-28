# USIS — Universal Spectral Intelligence System

Một **dụng cụ nghiên cứu** cho quang phổ học quang học (Raman + IR trước), đặt nền trên vật lý và
có ý thức về bất định. Phổ được coi là **phép đo vật lý của vật chất**; việc xác định thứ đã tạo ra
nó là một **bài toán ngược đặt không chỉnh**, mà đáp án là một hậu nghiệm cộng với một khối lượng
"không phải cái nào ở đây" được nêu tường minh.

> We are not building an AI that reads spectra. We are building an AI that knows how to do science
> with spectra — one that represents what it does not know, and chooses the measurement that would
> resolve it.


## Bắt đầu nhanh

```bash
uv sync
uv run usis --help
uv run usis info                     # phiên bản, git SHA, môi trường — đơn vị để replay
uv run usis licences                 # từ vựng giấy phép và cổng chặn vendor
uv run usis convert axis --from wavelength_nm --to wavenumber_cm-1 532
uv run usis methods                  # registry phương pháp: mức độ chín, giả định, bằng chứng
```

Cột `surfaces` của `usis methods` là cột nên đọc: nó trả lời "output của phương pháp này có được
xuất hiện trong một kết luận không?". Một dòng `hypothesized` với `NO` ở đó là shadow mode đang
hiển thị ra, chứ không phải được hàm ý.

## Phát triển

```bash
uv run pytest -q                                  # test (offline theo thiết kế)
uv run ruff check . && uv run ruff format --check .
uv run mypy                                       # strict, trên core/data/eval/cli/measurement/methods
uv build
```

## Cái gì nằm ở đâu

| Đường dẫn | Nội dung |
| --- | --- |
| `docs/` | Tài liệu thiết kế, ADR và ghi chú lĩnh vực |
| `usis/` | Mã nguồn của hệ thống |
| `tests/` | Bộ kiểm thử và fixture nhỏ được ghi nguồn trong từng thư mục |

## Những quy ước không thương lượng

- **Không có đáp án trần.** Output định danh là một `BeliefState`; `sum(posterior) + p_none == 1`
  được assert trong code.
- **Từ chối trả lời là một thành công.** "Không đủ bằng chứng, và đây là phép đo sẽ gỡ được nó" là
  một output hạng nhất.
- **Đơn vị có kiểu.** Raman shift không phải wavenumber. Phép chuyển đổi cần bước sóng kích thích
  thì raise chứ không đoán.
- **Provenance bất biến.** Xử lý được ghi lại, không bao giờ bị nướng thầm vào dữ liệu.
- **Không bịa gì cả.** Sự kiện chưa xác minh được đánh dấu `UNVERIFIED` và chặn mọi thứ phụ thuộc
  vào nó.

## Giấy phép

Apache-2.0 cho phần code. Giấy phép dữ liệu được ghi theo từng dataset trong `usis/data/licences.py`
và `docs/knowledge/datasets.md`; cổng CI từ chối vendor bất cứ thứ gì chưa xác minh hoặc không cho
phân phối lại.
