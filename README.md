# sekai-card-id

从用户截图里识别 Project SEKAI 卡牌，用于卡牌导入。

旧方案（对每张卡裁一块 52×52 做 64 位 pHash）已经移除，见 git 历史。新方案把识别做成**检索**：

```
卡片截图 ─► 裁出卡面区域、遮掉角标 ─► 冻结的预训练模型 ─► 向量 ─► 底库余弦 top-k ─► 像素级精排 ─► cardId + 特训状态
                                                                   ▲
                   官方缩略图 (thumbnail/chara_rip) ─► 同样的预处理和模型 ─┘
```

- **出新卡不需要训练**：CI 下载新卡缩略图、算向量、追加进底库，已有卡直接复用。
- 模型默认使用 DINOv2 ViT-S/14（`vit_small_patch14_dinov2.lvd142m`，冻结、不训练）。任何 timm 模型都可以换进来对比。
- 精排：对 top-k 候选，用 32×32 彩色小图做归一化互相关（NCC），并在 ±3% 平移、±3% 缩放范围内取最大值。用于区分同一角色的相似卡面，也能吸收裁剪没对准的误差。

目前实现了三部分：底库构建、零训练识别，以及游戏内卡牌列表（控え室）截图的卡片定位。

**卡片定位**不需要模型：列表里每张完整显示的卡，底部都有一条颜色固定的深色 `Lv.xx` 条，横跨整张卡宽。定位器先按颜色找到这些条，再按卡框配置推算出卡片框，最后用网格规律统一尺寸。只露出一部分、`Lv` 条不可见的卡会被跳过；它们会出现在下一张截图里，跨截图重复的卡按卡牌 ID 合并（同一张卡每个玩家最多持有一张）。

## 安装

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # 有 GPU 的话换成对应的源
pip install -e ".[ml,dev]"
```

模型权重从 Hugging Face 下载。国内网络可以设置 `HF_ENDPOINT=https://hf-mirror.com`，也可以用 `--weights` 指定本地权重文件（支持 `.pth`、`.safetensors`，以及 big_vision 的 `.npz`）。

## 用法

```bash
# 1. 拉取主数据和缩略图（增量：已下载的会跳过；缺失的会记录并在下次重试）
sekai-card-id sync --asset-dir data/assets \
  --asset-url 'https://storage.sekai.best/sekai-jp-assets/thumbnail/chara_rip/{bundle}_{state}.{ext}' \
  --ext webp,png

# 2. 构建 / 增量更新底库
sekai-card-id build --asset-dir data/assets --gallery data/gallery

# 1 + 2 合在一起（CI 跑的就是这个）
sekai-card-id update

# 整张截图 → 持有的卡牌（跨截图去重）。--debug-dir 会输出带标注的截图、每张卡的裁剪图，
# 以及预填了识别结果的 labels.csv，改掉识别错的行，就能直接用作评测集或校准数据
sekai-card-id scan shot1.png shot2.png --debug-dir out/

# 只做定位、导出裁剪图（不需要底库）
sekai-card-id detect shot1.png --out-dir crops/

# 识别单张卡片裁剪图，可以用已知信息缩小候选范围
sekai-card-id match crop1.png crop2.png --filter rarity=rarity_4

# 评测：没有真实截图时，用缩略图合成模拟截图
sekai-card-id eval --samples 1000 --jitter 0.03
# 评测：真实的卡片裁剪图（CSV 列为 path,card_id[,state]，path 相对于 CSV 所在目录）
sekai-card-id eval --labels out/labels.csv --out report.json

# 用带标注的裁剪图，自动拟合卡面在卡片中的位置
sekai-card-id calibrate --labels out/labels.csv --out my_layout.json
```

`eval` 会同时报告"只用向量"（`rerank_weight=0.0`）和"向量 + 精排"两组结果，其中 `errors_same_character` 是错成同一角色其他卡的数量。

`--model tiny` 是一个不需要下载的手工特征，只用于测试，以及作为对比的下限。

### 底库格式（`data/gallery/`）

| 文件 | 内容 |
|---|---|
| `meta.json` | 模型、layout、预处理版本的指纹，以及条目数和维度 |
| `entries.json` | 每行一个条目：`card_id`、`state`、`character_id`、`rarity`、`attr`、`release_at`、素材的 `sha256` |
| `embeddings.npy` | float16 (N, D)，已做 L2 归一化 |
| `templates.npy` | uint8 (N, 32, 32, 3)，用于精排的彩色小图 |

指纹不变时，按素材的 sha256 复用向量。更换模型或 layout 后，会自动全部重新计算。

## Layout：卡框样式变了只需要改这里

`sekai_card_id/layouts/default.json` 描述卡片的几何结构，全部是相对于卡片框的坐标：

- `card_to_art`：卡面图（正方形的 chara_rip）在卡片框里的位置；
- `card_masks`：卡片上被角标覆盖的区域（左上角属性图标；下方的星级、`Lv` 条、专精等级菱形角标）。这些区域在底库和查询两边都会被涂成同一种纯色，不参与比对。角标是相对卡框摆放的，所以用卡片坐标描述；校准 `card_to_art` 时不用改它们；
- `detector`：截图定位参数，包括 `Lv` 条的颜色和容差、条的高宽比、卡片框相对于 `Lv` 条的位置。这部分不影响底库，修改后不需要重建。

当前数值是在 2026-09 的控え室截图上量出来的（2000×923 的截图里，一张卡 130×130px）。**`card_to_art` 目前是假设卡面图一直延伸到 `Lv` 条下面**，还需要用真实素材确认：

1. `sekai-card-id scan shots/*.png --debug-dir out/`；
2. 检查 `out/labels.csv`，改正识别错的行，或者删掉不确定的行；
3. `sekai-card-id calibrate --labels out/labels.csv --out sekai_card_id/layouts/default.json`；
4. 修改 `card_to_art` 后需要重建底库：`sekai-card-id build --full`。

`legacy_156.json` 是旧版 156px 卡框的粗略估计，只作为参考保留。

## CI

- `.github/workflows/tests.yml`：运行单元测试，不需要 torch。
- `.github/workflows/gallery.yml`：每 6 小时运行一次 `update`。素材、底库和模型都会缓存，新卡只做增量计算；构建好的底库作为 artifact 上传。手动触发时可以勾选 `full`，全部重新计算。

## 现状与局限

- 默认素材地址（`storage.sekai.best`）没有经过验证，请换成你实际在用的镜像；主数据来自 `Sekai-World/sekai-master-db-diff`。
- 卡片定位已在 2 张真实截图上验证：80 张完整显示的卡全部找到，包括带专精等级角标的；只露出一部分的行被正确跳过。其他分辨率和屏幕比例的截图还没测过。
- 识别准确率还没有在真实卡面上评测过。开发环境无法访问素材 CDN 和 Hugging Face，只在代理数据上跑通了流程（见下）。
- 还没有拒识：底库里没有的卡也会被硬匹配成某张卡。`scan` 会输出 `score` 和 `margin`（第一名与第二名的分差），可以用真实数据定阈值。
- 如果列表滚动到一半，顶行卡片的上部被面板遮住，但 `Lv` 条仍然可见，这张卡仍会被检测出来，识别结果可能出错。
- 合成评测里的边框和角标是简化版，只能衡量模型对缩放、JPEG 压缩、偏移的鲁棒性，不能代替真实截图评测。

### 代理数据上的结果（仅用于验证流程）

测试数据有 294 张图：24 张照片，每张随机裁 6 个互相重叠的区域（模拟"同一角色的不同卡"这种难负例），再加 150 张程序生成的图片。查询图用 `eval` 合成，卡框配置用的是当时的占位值。模型用的是 ViT-S/16 AugReg in21k（DINOv2 在开发环境里下载不到）：

| 模型 | 偏移 jitter | 只用向量 top1 | 向量 + 精排 top1 | top5 |
|---|---|---|---|---|
| tiny（手工特征） | 0.03 | 0.993 | 0.997 | 1.000 |
| tiny（手工特征） | 0.06 | 0.922 | 0.932 | 0.997 |
| ViT-S/16 in21k | 0.03 | 0.752 | 0.993 | 1.000 |
| ViT-S/16 in21k | 0.06 | 0.694 | 0.959 | 0.993 |

结论：

1. 精排是必要的。ImageNet 监督训练的 ViT 学到的是语义特征，会把同一张照片的不同裁剪当成同一个东西，几乎所有错误都是错成同一组。
2. 这组数据的难度和真实卡面不同，不能用来选模型。选模型要在真实卡面和真实截图上用 `eval` 对比，候选包括 DINOv2、CLIP 和 tiny。

## 下一步

1. 用真实素材跑 `scan --debug-dir`，修正 `labels.csv`，然后 `calibrate`，再用 `eval --labels` 得到真实准确率，并对比 DINOv2、CLIP 和 tiny。
2. 用真实数据确定拒识阈值（`score` / `margin`）。
3. 如果零训练基线不够用，再做一次性的度量学习微调，并按卡牌上线时间切分来验证：新卡不需要重训。
4. 可选：识别 `Lv`、专精等级、特训状态等信息，一起导入。
