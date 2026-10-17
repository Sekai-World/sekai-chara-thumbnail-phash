Rolling release：`gallery` workflow 在底库内容有变化时自动更新，**不要手动编辑**。

- `manifest.json`：每个文件的 sha256、条目数、构建它的 commit。先读它，再下载文件并校验。
- `gallery-int8.tar.gz` + `embedder-int8.onnx`：int8 量化模型（约 24 MB）和对应的底库。
- `gallery-fp32.tar.gz` + `embedder-fp32.onnx`：未量化模型（约 88 MB）和对应的底库，识别结果与 PyTorch 完全一致。

同一个变体的底库和模型必须配套使用（用错了会报"embedder does not match"）。用法和注意事项见 README 的"发布的底库"一节。
