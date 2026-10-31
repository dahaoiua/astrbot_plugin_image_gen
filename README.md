# astrbot_plugin_image_gen

给 AstrBot 用的**通用绘图插件**：文生图（`generate_image`）、图生图（`edit_image`）、照图反推提示词（`describe_image_as_prompt`）。聊天时由 LLM 自己判断要不要画、要不要改，直接发到会话里。

"通用"指的是：

- **后端无关**：只要实现 `GET /healthz` 和 `GET /generate?... -> image/png` 这套简单协议，ComfyUI、SD WebUI、自建服务、云厂商 API 的包一层都能接
- **模型无关**：插件不假设工作流、模型名、采样器，只认接口
- **提示词中英混合**：中国/东亚元素保留中文原词，构图/光线/画质用英文 tag（见下）
- 默认配置已按本机 ComfyUI 调好，开箱即用

## 它是怎么"自己判断要不要画"的

两条路，按模型能力自动覆盖：

1. **工具调用（默认）**：注册 `generate_image` 工具，并在每次 LLM 请求前往 `system_prompt` 注入一段使用说明——
   什么场景该画、提示词该怎么写、什么情况不要画。支持 function calling 的模型会自己决定调用。
2. **备用标记（可选）**：给不支持工具调用的模型用。开启 `enable_marker` 后，模型只要在回复里另起一行写
   `[DRAW] 提示词`，插件会在发送前把这行摘掉、换成真实的图片。

## 提示词：中英混合

内置注入说明要求模型这样写提示词：

- **中国特有的元素保留中文**：人名地名、朝代、服饰、建筑、器物、纹样、画风，例如「水墨」「工笔」「敦煌壁画」「汉服」「唐宫」「龙纹」「牌坊」「江南水乡」
- **技术性描述用英文 tag**：构图、镜头、光线、材质、色调、画质，模型吃得更准
- 不要整段翻成英文，也别全中文

示例：

```
水墨风格，江南水乡的石拱桥与白墙黛瓦，薄雾，清晨斜射光,
traditional Chinese ink painting, misty morning light, ultra detailed, 8k
```

插件只负责在内置画质词后面追加、去重、按长度截断，提示词本身由 LLM 按上面的规则生成。

## 图生图（edit_image）

用户发图后说「改成…」「换成…」时，LLM 调 `edit_image`，插件把图 base64 传给后端 `edit_model`（默认 `qwen_lora_edit`）。

改历史图时传 `image_id`（即消息里的 `[图#12]`），插件从懒加载插件的库里取原图；`describe_image_as_prompt` 同理。

- 参考图来源：当前消息的图优先，没有就用引用消息的图
- 后端会先等比缩小参考图（长边 > 1024 才缩），出图尺寸跟随参考图
- 实测：768×1024 出图，本地约 40 秒，经隧道约 60 秒
## 反推提示词（照图作画）

工具 `describe_image_as_prompt`：用户说「照着这张图 / 上一条图画」时，LLM 先调它把图反推成提示词，再传给 `generate_image`。

- 取图：当前消息的图优先，没有就用引用消息的图
- 让视觉模型**直接输出成品提示词**（分层、权重、镜头词），不是先描述再改写，少一次信息损失
- 模型选择：`reverse_prompt_provider` 配置 > `provider_settings.default_image_caption_provider_id` > 当前对话模型
- 成本：deepseek-flash 走推理，一次约 2.7k token / 10~17 秒；换非推理 VLM 会快很多
## 提示词规范（注入给 LLM 的硬要求）

插件把下面这段塞进 `system_prompt`（约 10 行，纯给模型看，没有多余解释）：

```
【生图】可用 generate_image 工具画图。
- 触发：用户要图、或明显需要配图时调用；一条回复最多一张。
- 提示词结构（按序）：主体数量与身份, 外貌特征, 服装/状态, 动作/表情/互动, 环境/背景, 光影/氛围, 镜头/构图, 画质/风格。
- 短语逗号分隔，不写完整句子，40~120 词；不编造用户没提的细节。
- 关键特征加权：(词:1.2)，范围 0.5~1.5；弱化用 (词:0.8)。
- 必带镜头/视角/景深词：cowboy shot、from below:0.8、dutch angle、depth of field。
- 语言：英文。画面里要出现中文文字时，把中文原样写上。   ← krea2 时这一行；其它模型换成「中英混合，中国特有元素用中文原词，技术描述用英文。」
- negative_prompt 只放用户要求排除的元素。
- 不传尺寸；不改模型；出图后一两句话衔接，不复述提示词。
当前模型：Krea2 Turbo，768x1024
```

## 安装

```bash
cd AstrBot/data/plugins
git clone https://github.com/dahaoiua/astrbot_plugin_image_gen
# 重启 AstrBot 或到面板里重载插件
```

插件目录里**没有** `requirements.txt`，因为它只用标准库。

## 命令

| 命令 | 说明 |
| --- | --- |
| `/draw 一只在雪地里的白狐，胶片质感` | 手动出图，绕过 LLM |
| `/draw_model` | 看当前模型和可选项 |
| `/draw_model krea2` | 切换本会话的模型（`zimage` / `krea2` / `qwen_image`，`default` 恢复配置默认） |
| `/draw_size` | 看当前尺寸 |
| `/draw_size 1024x768` | 切换本会话的分辨率，也支持 `16:9`、`9:16`、`1024`（正方形） |
| `/draw_size default` | 恢复配置默认尺寸 |
| `/draw_status` | 检查出图接口是否在线，顺带显示引擎版本和显存 |

模型和尺寸按**会话**记忆（存在 `data/plugin_data/astrbot_plugin_image_gen/sessions.json`），
重启不掉；群里不同人/不同群可以各用各的。`generate_image` 工具不接收尺寸参数，模型即使传了 `width`/`height` 也会被忽略。

## 支持的模型（后端 profile）

插件本身不绑模型，模型由后端 `profile` 决定；本机这套后端内置五个：

| profile | 模型 | 默认步数 | 说明 |
| --- | --- | --- | --- |
| `zimage` | Z-Image Turbo | 8 | 最快，日常够用（不挂任何 LoRA） |
| `krea2` | Krea2 Turbo（12.6 GB）+ 影视柔光 LoRA 0.8 | 8 | 质感细腻、有柔光，靠分块换入所以慢 |
| `qwen_image` | Qwen-Image 2.1 | 25 | 官方 25 步版本，中文语义最好 |
| `qwen_turbo` | Qwen-Image 2.1 Turbo | 8（sigma 固定） | 蒸馏 UNet，比 25 步快 |
| `moss` | MOSS-Image + 4 步蒸馏 LoRA | 4 | **默认模型**；中文理解好，尺寸按 16 对齐 |
| `qwen_lora` | Qwen-Image 2.1 + 官方 turbo LoRA（文生图） | 8（sigma 固定） | 比 25 步版快，质量接近 |
| `qwen_lora_edit` | 同上 + 参考图（图生图） | 8（sigma 固定） | 发图让它改图；出图尺寸跟随参考图，长边超过 1024 自动缩小 |

> 显存只有 8 GB、且 TTS 常驻时三个都能跑，峰值都在 7.5 GB 左右；`krea2` 会把物理内存压到 1.8 GB 空闲，别同时开大程序。

## 配置

面板 → 插件 → 文生图 → 配置。所有项都有默认值，装完就能用。

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `api_base` | `http://127.0.0.1:8190` | 出图接口，换后端就改这里 |
| `api_timeout` | `300` | 接口超时（秒） |
| `api_user` / `api_password` | 空 | 出图接口在公网域名后时的 Basic Auth |
| `default_model` | `moss` | 默认模型；可填 `zimage`/`krea2`/`qwen_image`/`qwen_turbo`/`moss` |
| `default_size` | `768x1024` | 默认尺寸，支持 `1024x1024`、`16:9`、`1024` |
| `default_width` / `default_height` | `768` / `1024` | 仅当 `default_size` 留空时兜底 |
| `default_steps` | `0` | 0 = 用该模型自带步数；填了就全局覆盖 |
| `default_cfg` | `1.0` | CFG，见下面的注意事项 |
| `max_prompt_chars` | `1800` | 最终提示词长度上限（含画质词） |
| `quality_enabled` | `true` | 是否追加内置画质词 |
| `quality_positive` | 见 schema | 正向画质词 |
| `quality_negative` | 见 schema | 反向画质词 |
| `enable_tool` | `true` | 注册 `generate_image` 工具 |
| `inject_prompt` | `true` | 注入使用说明 |
| `inject_text` | 空 | 自定义注入文本，留空用内置 |
| `enable_marker` | `false` | 备用标记触发 |
| `marker_token` | `[DRAW]` | 标记文本 |
| `cooldown_seconds` | `15` | 同会话两次出图最小间隔 |
| `max_per_hour` | `30` | 同会话每小时上限 |
| `caption_enabled` / `caption_template` | `false` / `🎨 {prompt}` | 图片前附说明文字，可用 `{prompt} {seed} {width} {height} {duration}` |
| `keep_days` | `7` | 本地图片保留天数，0 不清理 |
| `allowed_sessions` / `blocked_sessions` | `[]` | 按 `unified_msg_origin` 部分匹配，留空表示不限 |

## 出图接口协议

插件只依赖三个端点：

```
GET <base>/healthz
GET <base>/profiles                       # 可选：列模型，拿不到就退化成"直接传 profile 字符串"
GET <base>/generate?prompt=...&profile=zimage&negative_prompt=...&width=512&height=512&steps=8&cfg=1&seed=...
```

`/generate` 默认返回 `image/png` 字节，响应头带 `X-Profile`、`X-Seed`、`X-Width`、`X-Height`、
`X-Prompt-Id`、`X-Duration`（没有也能跑）；带 `?format=json` 时返回同名字段组成的 JSON。

`profile` 不传就用后端默认；传了未知值后端返回 400。`steps` 不传则用该 profile 的默认步数
（zimage/krea2 8 步，qwen_image 25 步）。

默认实现是本机 8190 上的一个小服务（ComfyUI 包一层，约 380 行 Python），经 SSH 反向隧道映射到服务器
`127.0.0.1:8190`；插件不依赖它，换任何符合上面协议的服务都行。

## 注意事项

- **反向提示词大概率不起作用**：默认后端（本机 ComfyUI 的蒸馏工作流）靠 `cfg=1` 出图，
  此时 KSampler 不算 CFG，negative 不参与采样。实测把 `default_cfg` 提到 3 会明显崩图，所以默认保持 `1.0`，
  `quality_negative` 只作兼容保留。换成普通模型时可以把 cfg 调到 5~7，反向词就会生效。
  真要排除某类元素，写进正向提示词（例如 "clean background, no text"）更有效。
- **显存**：512×512 单张峰值约 7.5 GB / 8 GB（TTS 常驻时），已经比较紧；接口内部是串行的，
  不要并发调用，`max_per_hour` 也是为此准备的。
- **速度**：512×512、8 步，正常 10~15 秒一张；首次请求要加载模型，会多几秒。
- 生成的图存在 `data/plugin_data/astrbot_plugin_image_gen/images/`，按 `keep_days` 清理。
