# sekai-card-id

从用户截图里识别 Project SEKAI 卡牌，用于卡牌导入。

旧方案（对每张卡裁一块 52×52 做 64 位 pHash）已经移除，见 git 历史。新方案把识别做成**检索**：

```
卡片截图 ─► 裁出卡面区域、遮掉角标 ─► 冻结的预训练模型 ─► 向量 ─► 底库余弦 top-k ─► 像素级精排 ─► cardId + 特训状态
                                                                   ▲
                   官方缩略图 (thumbnail/chara) ─► 同样的预处理和模型 ─┘
```

- **出新卡不需要训练**：CI 下载新卡缩略图、算向量、追加进底库，已有卡直接复用。
- 模型默认使用 DINOv2 ViT-S/14（`vit_small_patch14_dinov2.lvd142m`，冻结、不训练）。任何 timm 模型都可以换进来对比。
- 精排：对 top-k 候选，用 32×32 彩色小图做归一化互相关（NCC），并在 ±3% 平移、±3% 缩放范围内取最大值。用于区分同一角色的相似卡面，也能吸收裁剪没对准的误差。

目前实现了三部分：底库构建、零训练识别，以及游戏内卡牌列表（控え室）截图的卡片定位。

**卡片定位**不需要模型：列表里每张完整显示的卡，底部都有一条颜色固定的深色 `Lv.xx` 条，横跨整张卡宽。定位器先按颜色找到这些条，再按卡框配置推算出卡片框，最后用网格规律统一尺寸。只露出一部分、`Lv` 条不可见的卡会被跳过；它们会出现在下一张截图里，跨截图重复的卡按卡牌 ID 合并（同一张卡每个玩家最多持有一张）。为了排除其他界面上同色的色块（例如横幅），每个候选条还要通过几条结构校验：上下两侧不是同色、左侧有白色的 `Lv.xx` 文字、卡片尺寸合理（不超过屏幕宽度的 30%），最后再按网格间距校验，偏离网格的丢弃。

**顶行被面板遮住**：列表滚动到一半时，顶行卡片的上部会渐隐到面板背景里，但 `Lv` 条仍然可见。被遮住的行显示的是面板背景，和卡片两侧空隙的背景一样平滑、颜色一致，所以对每张卡从上往下逐行比较，就能测出被遮住的比例 `clip_top`。遮住不超过 35%（`max_top_clip`）的卡只用可见部分识别：查询时向量和像素比对两边都忽略被遮住的区域；超过 35% 的卡直接跳过，它在相邻的截图里是完整的。

**拒识**：底库里没有的卡（比如新卡刚上线、底库还没更新）也总能找到一个"最像的"，所以需要判断要不要接受这个结果。依据两个指标：综合分 `score`，以及领先于"其他卡"最高分的差值 `margin`（同一张卡特训前后的另一张图不算竞争者）。阈值用 `eval --fit-reject` 拟合：把每条查询分别正常匹配一次，再把正确答案从底库里拿掉匹配一次（模拟底库里没有这张卡），在误收（接收了错误的卡，或接收了底库里没有的卡）不超过 `--target-error`（默认 1%）的前提下，让正确接收的数量最大。阈值存在底库目录的 `reject.json` 里，只对拟合时的模型和卡框配置有效。

`scan` 的输出分三类：
- `cards`：已接收的卡，按卡牌 ID 去重；
- `uncertain`：被拒识的，或者同一张截图里出现两次的同一张卡（其中必有一个是错的），附带前 3 个候选，方便界面让用户确认；
- `skipped`：被遮挡过多而跳过的卡。


## 安装

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # 有 GPU 的话换成对应的源
pip install -e ".[ml,dev]"
```

模型权重从 Hugging Face 下载。国内网络可以设置 `HF_ENDPOINT=https://hf-mirror.com`，也可以用 `--weights` 指定本地权重文件（支持 `.pth`、`.safetensors`，以及 big_vision 的 `.npz`）。

## 用法

```bash
# 1. 拉取主数据和缩略图（增量：已下载的会跳过；缺失的会记录并在下次重试）
sekai-card-id sync --asset-dir data/assets \
  --asset-url 'https://storage.sekai.best/sekai-jp-assets/thumbnail/chara/{bundle}_{state}.{ext}' \
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

# 拟合拒识阈值（写入 data/gallery/reject.json）。有真实标注时优先用 --labels
sekai-card-id eval --labels out/labels.csv --fit-reject --target-error 0.01

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
| `reject.json` | 可选，由 `eval --fit-reject` 生成的拒识阈值 |

指纹不变时，按素材的 sha256 复用向量。更换模型或 layout 后，会自动全部重新计算。

## Layout：卡框样式变了只需要改这里

`sekai_card_id/layouts/default.json` 描述卡片的几何结构，全部是相对于卡片框的坐标：

- `card_to_art`：卡面图（正方形的 thumbnail/chara 缩略图）在卡片框里的位置；
- `card_masks`：卡片上被角标覆盖的区域（左上角属性图标；下方的星级、`Lv` 条、专精等级菱形角标）。这些区域在底库和查询两边都会被涂成同一种纯色，不参与比对。角标是相对卡框摆放的，所以用卡片坐标描述；校准 `card_to_art` 时不用改它们；
- `detector`：截图定位参数，包括 `Lv` 条的颜色和容差、条的高宽比、卡片框相对于 `Lv` 条的位置。这部分不影响底库，修改后不需要重建。

当前数值来自 2026-09 的控え室截图（2000×923 的截图里，一张卡 130×130px），并已用 80 张核实过的真实卡片做过 `calibrate`：128px 的卡面缩略图**铺满整张卡片**，稀有度边框（上、右、下约 5px）、属性图标、星级、专精等级角标和 `Lv` 条都画在卡面之上，所以边框所在的细条也一并遮挡。校准后，裁剪图与官方卡面的平均相关系数从 0.796 提升到 0.990。

以后如果卡框样式又变了：

1. `sekai-card-id scan shots/*.png --debug-dir out/`；
2. 检查 `out/labels.csv`，改正识别错的行，或者删掉不确定的行；
3. `sekai-card-id calibrate --labels out/labels.csv --out sekai_card_id/layouts/default.json`；
4. 修改 `card_to_art` 或 `card_masks` 后需要重建底库：`sekai-card-id build --full`。

`legacy_156.json` 是旧版 156px 卡框的粗略估计，只作为参考保留。

## CI

- `.github/workflows/tests.yml`：运行单元测试，不需要 torch。
- `.github/workflows/gallery.yml`：每 6 小时依次运行 `sync` 和 `build`（即 `update`）。素材、底库和模型都会缓存，新卡只做增量计算。之后用 `tests/fixtures/screenshots/labels.csv` 里的 80 条真实标注重新拟合拒识阈值，并运行 `tests/test_real_recognition.py`，准确率下降时 CI 会报错。构建好的底库（含 `reject.json`）作为 artifact 上传。手动触发时可以勾选 `full`，全部重新计算。

本地运行真实识别测试：`SEKAI_GALLERY=data/gallery pytest tests/test_real_recognition.py`。

## 现状与局限

- 素材来自 `https://storage.sekai.best/sekai-jp-assets/thumbnail/chara/{bundle}_{state}.webp`（和 Sekai Viewer 相同，128×128），主数据来自 `Sekai-World/sekai-master-db-diff`。2026-09 时共 1452 张卡、2427 张卡面图，其中 7 张出厂就已特训的卡只有特训后的图。
- 测试样本（`tests/fixtures/screenshots/`）是 2 张 2000×923 的真实截图，以及 80 张卡的核实标注（`labels.csv`，按截图、行、列记录）。其他分辨率、屏幕比例的截图，以及真实的"滚到一半"截图还没有，欢迎补充。
- 顶行遮挡的测试用的是在真实截图上模拟的遮挡（按游戏的渐隐方式盖上面板背景）。
- 公开截图：在攻略站（AppMedia、Gamerch）的 70 张配图上验证过，没有误检；但其中没有可用的其他设备的控え室截图。reddit 对这个开发环境的访问会被重定向到登录页。

### 真实数据上的结果

底库：全部 2420 张卡面图，JP 服，2026-09。真实截图：上面 2 张，共 80 张卡。合成截图：用官方卡面按新版卡框渲染，加上缩放、JPEG 压缩、颜色抖动和裁剪偏移。

| | DINOv2 ViT-S/14（默认） | tiny（手工特征） |
|---|---|---|
| 真实截图 top1（含特训状态） | **80/80** | **80/80** |
| 合成截图，偏移 3%（接近真实定位精度） | 0.999 | 1.000 |
| 合成截图，偏移 6% | 0.985 | 0.987 |
| 合成截图，偏移 10% | **0.962** | 0.685 |
| 真实截图：正确卡最低分 / 未知卡最高分 | 0.954 / 0.864 | 0.989 / 0.885 |
| 每条查询耗时（CPU） | ~46 ms | ~0.7 ms |

- 实际定位误差约 1 像素（小于 1%），在这种条件下两种特征都几乎不会出错。卡片定位越不准，DINOv2 的优势越明显，所以默认仍然用 DINOv2。tiny 不需要下载模型，可以作为轻量方案（例如在浏览器里直接跑）。
- 像素精排在小偏移时能减少错误（DINOv2 在 3% 偏移下从 9 个错误降到 1 个），但它只搜索 ±3% 的范围，偏移很大时反而会拉低准确率。
- 拒识阈值（DINOv2，在 80 条真实标注上拟合）：`min_score=0.909`、`min_margin=0.113`。真实截图：80/80 接收且全部正确；80 张模拟的未知卡全部拒绝。用 1000 张更难的合成截图交叉检验：错误接收 0 张，未知卡误收 0/1000；有 87.8% 的已知卡被接收，其余转为让用户确认。只在合成数据上拟合的阈值偏松（合成截图比真实截图难，得分偏低），所以应当优先用真实标注拟合。

## 下一步

1. 收集其他设备（平板、不同屏幕比例）以及滚到一半的真实截图，补充到 `tests/fixtures/screenshots/` 和 `labels.csv`。样本越多，拒识阈值越可靠。
2. 其他服务器（EN/TW/KR/CN）：换对应的主数据和素材地址，界面相同的话卡框配置可以直接复用。
3. 可选：识别 `Lv`、专精等级等信息，一起导入。
4. 目前的零训练基线在真实数据上已经够用，暂时不需要微调。
