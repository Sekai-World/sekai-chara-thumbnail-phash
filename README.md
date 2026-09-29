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

目前实现的是第 1 步（底库构建）和第 2 步（零训练基线）。截图中的卡片定位（检测器）还没有做，所以 `match` 和 `eval` 的输入是**单张卡片的裁剪图**（带边框）。

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

# 识别单张卡片裁剪图，可以用已知信息缩小候选范围
sekai-card-id match crop1.png crop2.png --filter rarity=rarity_4

# 评测：没有真实截图时，用缩略图合成模拟截图
sekai-card-id eval --samples 1000 --jitter 0.03
# 评测：真实的卡片裁剪图（CSV 列为 path,card_id[,state]，path 相对于 CSV 所在目录）
sekai-card-id eval --labels real/labels.csv --out report.json
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

`sekai_card_id/layouts/default.json` 描述卡片裁剪图的几何结构（全部是相对坐标）：

- `card_to_art`：卡面图在卡片裁剪图中的位置（去掉边框后的区域）；
- `masks`：卡面上会被角标覆盖的区域（属性图标、星级、等级等）。这些区域在底库和查询两边都会被涂成同一种纯色，不参与比对。

**现在的数值是按旧版 156px 卡框估的占位值**，需要用新样式的真实截图校准：可以复制一份改成 `layouts/<name>.json`，然后用 `--layout <name>` 指定。改动 layout 后需要重新构建底库。

## CI

- `.github/workflows/tests.yml`：运行单元测试，不需要 torch。
- `.github/workflows/gallery.yml`：每 6 小时运行一次 `update`。素材、底库和模型都会缓存，新卡只做增量计算；构建好的底库作为 artifact 上传。手动触发时可以勾选 `full`，全部重新计算。

## 现状与局限

- 默认素材地址（`storage.sekai.best`）没有经过验证，请换成你实际在用的镜像；主数据来自 `Sekai-World/sekai-master-db-diff`。
- 目前还没有在真实卡面和真实截图上评测过。开发环境无法访问素材 CDN 和 Hugging Face，只在代理数据上跑通了流程（见下）。
- 合成评测里的边框和角标是简化版，只能衡量模型对缩放、JPEG 压缩、偏移的鲁棒性，不能代替真实截图评测。

### 代理数据上的结果（仅用于验证流程）

测试数据有 294 张图：24 张照片，每张随机裁 6 个互相重叠的区域（模拟"同一角色的不同卡"这种难负例），再加 150 张程序生成的图片。查询图用 `eval` 合成。模型用的是 ViT-S/16 AugReg in21k（DINOv2 在开发环境里下载不到）：

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

1. 收集几十张新样式的真实截图，校准 layout，并建立带标注的评测集（`eval --labels`）。
2. 截图中的卡片定位（单类别检测器，主要用合成截图训练）。
3. 如果零训练基线不够用，再做一次性的度量学习微调，并按卡牌上线时间切分来验证：新卡不需要重训。
