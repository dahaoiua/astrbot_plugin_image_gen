# -*- coding: utf-8 -*-
"""通用文生图插件（AstrBot）。

只做三件事：
1. 把「要不要画图」交给 LLM —— 注册 `generate_image` 工具，并在每次 LLM 请求前注入使用说明；
2. 真的去画 —— 调配置里的出图 API（默认本机 8190，走 SSH 反向隧道）；
3. 画完发到会话里 —— 图片存到 data/plugin_data/astrbot_plugin_image_gen/images/。

后端无关：只要实现 `GET /healthz` + `GET /generate?... -> image/png` 就行，换 ComfyUI 之外的引擎也能用。
另有一个备用触发方式（`enable_marker`）：模型不支持工具调用时，让它输出 `[DRAW] 提示词`，
插件在发送前把这一行摘掉、换成图片。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Plain
from astrbot.api.star import Context, Star, register
from astrbot.core.star.filter.command import GreedyStr
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

try:  # 正常加载时是包内相对导入
    from .core import (
        DEFAULT_NEGATIVE,
        DEFAULT_POSITIVE,
        GenResult,
        ImageApiClient,
        ImageApiError,
        compose_negative,
        compose_positive,
        parse_marker,
        parse_size,
    )
except ImportError:  # 直接按脚本方式加载时退化成绝对导入
    from core import (  # type: ignore
        DEFAULT_NEGATIVE,
        DEFAULT_POSITIVE,
        GenResult,
        ImageApiClient,
        ImageApiError,
        compose_negative,
        compose_positive,
        parse_marker,
        parse_size,
    )

PLUGIN_NAME = "astrbot_plugin_image_gen"

# 注入给 LLM 的使用说明。{lang} 按当前模型替换成唯一一条语言规则。
# 文生图 / 图生图各有一套写法（可在配置里分别替换 prompt_t2i / prompt_edit）。
INJECT_COMMON = """【生图】可用 generate_image（文生图）和 edit_image（图生图改图）；一条回复最多一张。
- 触发：用户要图、或明显需要配图时用 generate_image；用户发图后说"改成…""换成…""加个…"时用 edit_image。
- 用户要「照着这张图 / 上一条图」画时，先调 describe_image_as_prompt 反推提示词，再把结果传给 generate_image；普通描述画图别多这一步。
- 消息里的 [图#12] 表示一张存下来的图：要改它/照着它反推时，把 12 传给对应工具的 image_id。
- 用户说「刚才那张」「这张图」或者直接下调图指令（加景深/压暗角/换背景/增强质感）时：调 edit_image 且**不要传 image_id**（默认改最近一张，也就是你刚发出去的那张）；只有用户明确指向更早的某张图时才用编号。
- 分辨率只限长边（默认 1024），比例不进提示词，走 aspect 参数：
  · 文生图（generate_image）：比例由你按画面决定，3:4 竖构图 / 2:3 人像 / 1:1 方图 / 3:2 横构图 / 16:9 宽屏。
  · 图生图（edit_image）：**默认保留原图比例**，不要传 aspect；只有用户明确要换比例（改成横版/竖版/方图）时才传。
- 用户说「放大/超分/高清一点/4K/清晰点」时用 upscale_image（默认 4 倍，可传 scale），默认用最近一张或 [图#N]。
- 画知名 IP 角色时（原神/星铁/初音/Fate 等）：提示词里角色部分**只写标准标签组合**「角色名 (作品名)」，例如 furina (genshin impact)、nahida (genshin impact)、hatsune miku；**不要自己描述外貌**，除非用户明确要求改外貌/服装。
- IP 形象校对（配置开着时）：画 IP 角色调 generate_image 要传 ip_character=true，出图后**不会直接发**；先 view_image(编号) 看一眼，像就 send_image(编号) 发出去，不像才补外貌描述重画（最多重画一次），画对了再发。
- negative_prompt 只放用户要求排除的元素。
- 不传尺寸；不改模型；出图后一两句话衔接，不复述提示词。"""

T2I_SPEC = """【文生图提示词】（Qwen-Image 官方改写规范，压缩版）
你是在描述一张已经画好的图，不是在下指令：英文、现在时、第三人称，一次输出**一整段**连贯描述（约 20 句、400~500 词），短要求不等于写短。

- 开头句（约 20 词，媒介不能省）：The image is a ⟨vertical/wide/square⟩ ⟨style⟩ ⟨photograph/poster/illustration/...⟩ of ⟨subject⟩, ⟨background and palette⟩.
- 用户明确给的文字、对象、数量、颜色、位置必须原样保留；没提到的背景、光线、氛围由你补全并决定。
- 走法二选一：主体填满画面 → 背景与虚化 → 姿态位置 → 头部面部 → 身体与衣物 → 手持物 → 边缘；画面分区（海报/版面/多区域）→ 背景 → 顶部 → 左中右 → 底部。
- 约三分之一的句子用位置短语开头（On the right side of the frame, ... / In the upper-left corner, ...），让画面始终可定位。
- 画面里的文字逐字写进双引号，交代位置、字重、颜色、大小；读不清的写 blurred / too small to read，不要编字。
- 光影单独成句：光源、方向、软硬、以及留下的高光与影子（The lighting is ...）。默认写定向硬光（hard key light / crisp-edged shadows），除非用户要柔光、阴天、平光。
- 结尾一句收整体：构图、配色、风格、氛围（The overall composition ...），只写这一句。
- 质感要求：颜色带修饰词（deep navy、muted olive），材质写具体（brushed metal、coarse linen、weathered wood），枚举不概括，人物写可观察的外表，物体说类别不说品牌。
- 禁止：分辨率、比例、masterpiece/8K 这类质量词、任何 "you/create/make sure" 式指令；拿不准的地方用 appears to be / likely 这种观察者口吻。
- 语言：{lang}"""

EDIT_SPEC = """【图生图提示词】（Qwen-Image 官方编辑规范，压缩版）
写成一条可执行的编辑指令：英文、一段到底、动词开头，只输出指令本身。

- 只改用户点名的那几个属性，而且改到位（弱到看不出变化等于没改）。
- 其余一律保持：用**一句**总的保留句带过（keep everything else unchanged），不要逐个描写要保留的内容——写得越细越容易被改掉。
- 身份是不变量：人物脸与可辨识配饰、产品外形与标记、原图的媒介风格（照片/动漫/插画/3D），除非用户明确要改。
- 顺序：先说要改什么、改在哪里，再一句保留；不要先描述成品画面。
- 文字逐字写进双引号；拿不准的字宁可不加。
- 改光影写清方向、软硬和影子变化：relight as warm side light from camera-right, long shadows to the left, keep a rim light on her shoulder。
- 多张参考图时按顺序写 first image / second image；不要用"图1/第一张"这种说法。
- 禁止：分辨率、比例、质量词。
- 语言：{lang}"""



def _lazy_store_dir(umo: str) -> Path:
    base = Path(get_astrbot_plugin_data_path()) / "astrbot_plugin_lazy_image"
    key = hashlib.sha1((umo or "default").encode("utf-8")).hexdigest()[:16]
    return base / key


def _lazy_latest_path(umo: str, uid: str = "") -> str:
    """懒加载库里这个人的「刚才那张」：在他发的图和回给他的图里挑最新一张。"""
    sdir = _lazy_store_dir(umo)
    idx = sdir / "index.json"
    if not idx.is_file():
        return ""
    try:
        data = json.loads(idx.read_text(encoding="utf-8"))
    except Exception:
        return ""
    items = data.get("items") or []
    if not items:
        return ""
    picked = items[-1]
    if uid:
        mine = [it for it in items if str(it.get("uid") or "") == str(uid)]
        if mine:
            picked = mine[-1]
    path = sdir / str(picked.get("file") or "")
    return str(path) if path.is_file() else ""


def _lazy_register(umo: str, path: str, width: int = 0, height: int = 0, uid: str = "") -> int:
    """把刚出的图登记进懒加载库，这样「刚才那张」有编号可用。返回编号，失败返回 0。"""
    src = Path(path)
    raw = src.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    sdir = _lazy_store_dir(umo)
    sdir.mkdir(parents=True, exist_ok=True)
    idx = sdir / "index.json"
    data = {"last_id": 0, "items": []}
    if idx.is_file():
        try:
            data = json.loads(idx.read_text(encoding="utf-8"))
        except Exception:
            data = {"last_id": 0, "items": []}
    for item in data.get("items", []):
        if item.get("sha") == sha:
            return int(item.get("id", 0))
    new_id = int(data.get("last_id", 0)) + 1
    name = f"{new_id}{src.suffix or '.png'}"
    (sdir / name).write_bytes(raw)
    data["last_id"] = new_id
    data.setdefault("items", []).append({
        "id": new_id, "file": name, "sha": sha,
        "w": int(width or 0), "h": int(height or 0),
        "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "src": "bot", "uid": str(uid or ""),
    })
    idx.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return new_id


def _lazy_ref_path(umo: str, image_id: int) -> str:
    """按编号从懒加载插件的库里取图（同一个 plugin_data 根目录）。"""
    base = Path(get_astrbot_plugin_data_path()) / "astrbot_plugin_lazy_image"
    key = hashlib.sha1((umo or "default").encode("utf-8")).hexdigest()[:16]
    idx = base / key / "index.json"
    if not idx.is_file():
        return ""
    try:
        data = json.loads(idx.read_text(encoding="utf-8"))
    except Exception:
        return ""
    for item in data.get("items", []):
        try:
            if int(item.get("id", -1)) == int(image_id):
                path = base / key / str(item.get("file") or "")
                return str(path) if path.is_file() else ""
        except (TypeError, ValueError):
            continue
    return ""


MODEL_LABELS = {
    "zimage": "Z-Image Turbo",
    "krea2": "Krea2 Turbo",
    "qwen_image": "Qwen-Image 2.1",
}

# 语言规则：默认全英文，只有中国特有元素和画面里要原样出现的中文才用中文
LANG_RULE = "全英文。只有两种情况用中文：①中国特有的元素（人名地名、朝代、服饰、建筑、纹样、画风）；②画面里要原样出现的中文文字。"

# 反推提示词：让视觉模型直接产出「能直接生图的提示词」，不要先描述再改写
REVERSE_INSTRUCTION = """你是文生图提示词工程师。看清楚这张图，用一段连贯的英文把画面写成能直接送进文生图模型的提示词——像在描述一张已经画好的成品（Qwen-Image 官方写法）。

只输出提示词本身，不要前言、不要"这是一张…"这类解释。

要求：
- 开头句约 20 词：The image is a [vertical/wide/square] [realistic/cinematic/minimalist/...] [photograph/poster/illustration/...] of [subject], [background and palette].
- 之后按序展开：medium → subject → action → relationships → composition → lighting → materials → exact lettering
- 图里出现的具体文字用引号括起来并交代位置
- 光影像照片一样交代清楚：光源从哪来、软硬和色温、影子落在哪里（例如 hard window light from the left, long shadows across the floor）
- 不要堆关键词，不要写尺寸或分辨率
- 语言：{lang}"""



@dataclass
class _GenOutcome:
    path: Path
    result: GenResult
    prompt: str


@register(
    PLUGIN_NAME,
    "dahaoiua",
    "让 LLM 自己决定要不要画图：调可配置的出图接口（默认本机 ComfyUI），自带中英混合提示词、质量词与一堆可调项",
    "v1.0.0",
)
class ImageGenPlugin(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context, config)
        self.config = config or {}
        base = Path(get_astrbot_plugin_data_path()) / PLUGIN_NAME
        self.image_dir = base / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self._last_at: dict[str, float] = {}
        self._hits: dict[str, list[float]] = {}
        self._last_cleanup = 0.0
        self._overrides_path = base / "sessions.json"
        self._overrides: dict[str, dict] = self._load_overrides()

        logger.info(
            "[imggen] 已加载：api=%s 工具=%s 注入=%s 备用标记=%s",
            self._str("api_base", ""),
            self._bool("enable_tool", True),
            self._bool("inject_prompt", True),
            self._bool("enable_marker", False),
        )

    # ------------------------------------------------------------- 配置读取
    def _raw(self, key: str, default):
        try:
            value = self.config.get(key, default)
        except Exception:  # pragma: no cover - 配置对象异常时兜底
            return default
        return default if value is None else value

    def _bool(self, key: str, default: bool) -> bool:
        value = self._raw(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "是", "开"}
        return bool(value)

    def _int(self, key: str, default: int) -> int:
        try:
            return int(float(str(self._raw(key, default)).strip()))
        except (TypeError, ValueError):
            return default

    def _float(self, key: str, default: float) -> float:
        try:
            return float(str(self._raw(key, default)).strip())
        except (TypeError, ValueError):
            return default

    def _str(self, key: str, default: str) -> str:
        value = self._raw(key, default)
        return default if value is None else str(value)

    def _list(self, key: str, default: list | None = None) -> list:
        value = self._raw(key, default or [])
        if isinstance(value, (list, tuple)):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, str):
            return [x.strip() for x in value.replace("\n", ",").split(",") if x.strip()]
        return list(default or [])

    def _client(self) -> ImageApiClient:
        return ImageApiClient(
            base_url=self._str("api_base", "http://127.0.0.1:8190"),
            timeout=self._int("api_timeout", 300),
            user=self._str("api_user", ""),
            password=self._str("api_password", ""),
        )

    # --------------------------------------------------------------- 会话判断
    def _session_allowed(self, event: AstrMessageEvent) -> bool:
        umo = str(getattr(event, "unified_msg_origin", "") or "")
        blocked = self._list("blocked_sessions")
        if blocked and any(item in umo for item in blocked):
            return False
        allowed = self._list("allowed_sessions")
        if allowed and not any(item in umo for item in allowed):
            return False
        return True

    def _rate_limited(self, event: AstrMessageEvent) -> str | None:
        umo = str(getattr(event, "unified_msg_origin", "") or "")
        now = time.time()
        cooldown = self._int("cooldown_seconds", 15)
        if cooldown > 0:
            last = self._last_at.get(umo, 0.0)
            if now - last < cooldown:
                return f"刚画过一张，{int(cooldown - (now - last)) + 1} 秒后再试。"
        limit = self._int("max_per_hour", 30)
        if limit > 0:
            hits = [t for t in self._hits.get(umo, []) if now - t < 3600]
            self._hits[umo] = hits
            if len(hits) >= limit:
                return "这个会话一小时内画得太多了，先歇一会儿。"
        return None

    def _mark_used(self, event: AstrMessageEvent) -> None:
        umo = str(getattr(event, "unified_msg_origin", "") or "")
        now = time.time()
        self._last_at[umo] = now
        self._hits.setdefault(umo, []).append(now)

    # ------------------------------------------------------------ 会话级设置
    @staticmethod
    def _session_key(event: AstrMessageEvent) -> str:
        return str(getattr(event, "unified_msg_origin", "") or "")

    def _load_overrides(self) -> dict:
        try:
            data = json.loads(self._overrides_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_overrides(self) -> None:
        try:
            self._overrides_path.write_text(
                json.dumps(self._overrides, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:  # pragma: no cover
            logger.warning("[imggen] 保存会话设置失败", exc_info=True)

    def _default_long_side(self) -> int:
        raw = self._str("default_size", "").strip()
        if raw:
            w, h = parse_size(raw)
            return max(int(w), int(h))
        return max(self._int("default_width", 1024), self._int("default_height", 1024), 256)

    def _current_model(self, event: AstrMessageEvent) -> str:
        ov = self._overrides.get(self._session_key(event)) or {}
        return str(ov.get("model") or self._str("default_model", "")).strip().lower()

    def _current_long_side(self, event: AstrMessageEvent) -> int:
        """会话的长边设置；比例由模型自己定，这里只管长边。"""
        ov = self._overrides.get(self._session_key(event)) or {}
        raw = str(ov.get("long_side") or "").strip()
        if raw.isdigit():
            return max(256, int(raw))
        return self._default_long_side()

    def _set_override(self, event: AstrMessageEvent, **kwargs) -> None:
        key = self._session_key(event)
        cur = dict(self._overrides.get(key) or {})
        for k, v in kwargs.items():
            if v in (None, "", "default", "reset", "默认"):
                cur.pop(k, None)
            else:
                cur[k] = v
        if cur:
            self._overrides[key] = cur
        else:
            self._overrides.pop(key, None)
        self._save_overrides()

    # ----------------------------------------------------------------- 出图
    async def _generate(
        self,
        event: AstrMessageEvent,
        prompt: str,
        *,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        negative: str | None = None,
        model: str | None = None,
        size: tuple[int, int] | None = None,
        image: str = "",
        aspect: str = "",
        long_side: int | None = None,
    ) -> _GenOutcome:
        if not self._session_allowed(event):
            raise ImageApiError("这个会话没有开生图权限。")
        limited = self._rate_limited(event)
        if limited:
            raise ImageApiError(limited)

        quality_on = self._bool("quality_enabled", True)
        final_prompt = compose_positive(
            prompt,
            self._str("quality_positive", DEFAULT_POSITIVE),
            quality_on,
            self._int("max_prompt_chars", 1800),
        )
        if not final_prompt:
            raise ImageApiError("提示词是空的。")
        final_negative = compose_negative(
            negative or "",
            self._str("quality_negative", DEFAULT_NEGATIVE),
            quality_on,
        )

        # 只限长边，比例交给模型/参数决定
        side = int(long_side) if long_side else self._current_long_side(event)
        if size:
            side = max(int(size[0]), int(size[1]))
        aspect_value = (aspect or "").strip().lower()
        width = height = side
        if steps is None:
            configured = self._int("default_steps", 0)
            steps = configured if configured > 0 else None
        # default_cfg 填 0 = 用后端 profile 自己的 cfg（moss 是 2.5，turbo 系是 1）
        if cfg is None:
            configured_cfg = self._float("default_cfg", 0.0)
            cfg_value = configured_cfg if configured_cfg > 0 else None
        else:
            cfg_value = float(cfg)
        profile = (model or self._current_model(event) or "").strip().lower()

        client = self._client()
        timeout = self._int("api_timeout", 300)
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    client.generate,
                    final_prompt,
                    final_negative,
                    width,
                    height,
                    steps,
                    cfg_value,
                    None,
                    profile,
                    image,
                    aspect_value,
                    side,
                ),
                timeout=timeout + 15,
            )
        except asyncio.TimeoutError as exc:
            raise ImageApiError(f"出图超时（>{timeout}s）。") from exc

        self._mark_used(event)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.image_dir / f"{stamp}_{result.seed}.png"
        path.write_bytes(result.image)
        try:
            img_id = _lazy_register(
                getattr(event, "unified_msg_origin", "") or "",
                str(path), result.width or width, result.height or height,
                str(getattr(event, "get_sender_id", lambda: "")() or ""),
            )
            if img_id:
                logger.info("[imggen] 出图登记为 #%s（之后「刚才那张」用它）", img_id)
        except Exception:
            logger.debug("[imggen] 登记到懒加载库失败", exc_info=True)
        logger.info(
            "[imggen] 出图完成 %s %sx%s seed=%s %.1fs -> %s",
            result.profile or profile or "default", width, height, result.seed,
            result.duration, path.name,
        )
        self._cleanup_images()
        return _GenOutcome(path=path, result=result, prompt=final_prompt)

    def _cleanup_images(self) -> None:
        keep_days = self._int("keep_days", 7)
        if keep_days <= 0:
            return
        now = time.time()
        if now - self._last_cleanup < 3600:
            return
        self._last_cleanup = now
        cutoff = now - keep_days * 86400
        try:
            for item in self.image_dir.glob("*.png"):
                if item.stat().st_mtime < cutoff:
                    item.unlink(missing_ok=True)
        except OSError:  # pragma: no cover
            pass

    def _image_components(self, outcome: _GenOutcome) -> list:
        """按配置拼「图片(+可选说明文字)」的组件列表。"""
        comps: list = []
        if self._bool("caption_enabled", False):
            template = self._str("caption_template", "🎨 {prompt}")
            try:
                caption = template.format(
                    prompt=outcome.prompt,
                    seed=outcome.result.seed,
                    width=outcome.result.width,
                    height=outcome.result.height,
                    duration=f"{outcome.result.duration:.1f}",
                )
            except (KeyError, IndexError, ValueError):
                caption = template
            if caption.strip():
                comps.append(Plain(caption))
        comps.append(Image.fromFileSystem(str(outcome.path)))
        return comps

    # ------------------------------------------------------------------ 工具
    @filter.llm_tool(name="generate_image")
    async def generate_image_tool(
        self,
        event: AstrMessageEvent,
        prompt: str,
        negative_prompt: str | None = None,
        aspect: str = "",
        ip_character: bool = False,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
    ) -> str:
        """根据文字描述用配置好的出图服务生成一张图片，并直接发给用户。

        Args:
            prompt(string): 按官方规范写的提示词（英文一段连贯描述，观察者视角）
            negative_prompt(string): 可选，需要避免出现在画面里的元素
            aspect(string): 可选，画面比例，按内容自己选：3:4 竖构图 / 2:3 人像 / 1:1 方图 / 3:2 横构图 / 16:9 宽屏；不传默认 3:4
            ip_character(bool): 画知名 IP 作品里的角色（原神/星铁/初音等）时传 true；开了形象校对就先不发送，交给 send_image
        """
        if not self._bool("enable_tool", True):
            return "生图功能当前已关闭，直接回答用户即可。"
        # 兼容：老提示词/模型习惯会带 width/height/steps，一律忽略，尺寸只认 /draw_size 和配置
        if width or height or steps:
            logger.debug(
                "[imggen] 忽略模型给的尺寸/步数 width=%s height=%s steps=%s（由 /draw_size 控制）",
                width, height, steps,
            )
        try:
            outcome = await self._generate(
                event,
                prompt,
                negative=negative_prompt,
                aspect=aspect,
            )
        except ImageApiError as exc:
            logger.warning("[imggen] 工具出图失败：%s", exc)
            reason = str(exc).strip().rstrip("。.")
            return f"生图失败：{reason}。可以直接告诉用户这次没画成。"
        except Exception as exc:  # pragma: no cover - 防御性兜底
            logger.exception("[imggen] 工具出图异常")
            return f"生图异常：{exc}"

        if ip_character and self._bool("ip_check_enabled", True):
            # 形象校对：先不发，存进图库，交给模型看一眼再决定
            iid = _lazy_register(
                getattr(event, "unified_msg_origin", "") or "", str(outcome.path),
                outcome.result.width, outcome.result.height,
                str(getattr(event, "get_sender_id", lambda: "")() or ""),
            )
            logger.info("[imggen] IP 形象校对：先不发送，图#%s 等确认", iid)
            return (
                f"已生成但**没有发送**：图#{iid}（{outcome.result.width}x{outcome.result.height}）。\n"
                f"先调 view_image({iid}) 看一眼：像这个角色就调 send_image({iid}) 发出去；"
                "不像就补上外貌/服装描述重新 generate_image（ip_character=true），画对了再发。"
            )

        try:
            await event.send(event.chain_result(self._image_components(outcome)))
        except Exception:  # pragma: no cover - 发图失败不影响文字回复
            logger.exception("[imggen] 发送图片失败")
            return "图片已生成但发送失败了，告诉用户稍后再试。"
        return (
            f"已生成并发送图片（{outcome.result.width}x{outcome.result.height}，"
            f"seed={outcome.result.seed}，耗时 {outcome.result.duration:.1f}s）。"
            "不要重复调用工具，用一两句话自然衔接即可。"
        )

    @filter.llm_tool(name="send_image")
    async def send_image_tool(self, event: AstrMessageEvent, image_id: int = 0) -> str:
        """把已经生成但还没发出去的图发给用户（IP 形象校对通过后用它发送）。

        Args:
            image_id(int): 要发的图编号，比如 [图#12] 的 12；不传就发最近一张
        """
        umo = getattr(event, "unified_msg_origin", "") or ""
        path = _lazy_ref_path(umo, image_id) if image_id else ""
        if not path:
            path = _lazy_latest_path(umo, str(getattr(event, "get_sender_id", lambda: "")() or ""))
        if not path:
            return "没找到这张图（可能已被清理），重新 generate_image 一张。"
        try:
            await event.send(event.chain_result([Image.fromFileSystem(path)]))
        except Exception:  # pragma: no cover
            logger.exception("[imggen] 发送图片失败")
            return "发送失败，告诉用户稍后再试。"
        return "已发送。用一两句话自然衔接即可，不要重复调用。"

    # ------------------------------------------------------ 图生图：改一张图
    @filter.llm_tool(name="edit_image")
    async def edit_image_tool(
        self,
        event: AstrMessageEvent,
        prompt: str,
        negative_prompt: str | None = None,
        image_id: int = 0,
        aspect: str = "",
    ) -> str:
        """按用户的话改一张图（图生图）：用户发图后说「改成…」「换成…」「加个…」时用。

        Args:
            prompt(string): 具体怎么改，例如"把背景换成夜晚的街道""把裙子改成红色""给她加一顶帽子"
            negative_prompt(string): 可选，不要出现的元素
            image_id(int): 可选，消息里 [图#12] 的编号；改旧图时用它，不传就用当前消息/引用的图
            aspect(string): 可选，输出比例。默认不传=保留原图比例；只有用户明确要换比例（横版/竖版/方图）时才传，例如 16:9 / 1:1 / 3:4
        """
        if not self._bool("enable_tool", True):
            return "生图功能当前已关闭，直接回答用户即可。"
        umo = getattr(event, "unified_msg_origin", "") or ""
        refs = []
        if image_id:
            picked = _lazy_ref_path(umo, image_id)
            refs = [picked] if picked else []
        if not refs:
            refs = await self._collect_reference_images(event)
        if not refs:
            # 「刚才那张」「给它加景深」这类没带图也没编号的调图指令：默认改最近一张
            latest = _lazy_latest_path(umo, str(getattr(event, "get_sender_id", lambda: "")() or ""))
            if latest:
                refs = [latest]
                logger.info("[imggen] 没指定图片，用最近一张 %s", Path(latest).name)
        if not refs:
            return "没找到要改的图：让用户先发一张图，或者引用那张图再试。"
        src = Path(refs[0])
        try:
            raw = src.read_bytes()
        except OSError as exc:
            return f"读不到参考图：{exc}"
        suffix = src.suffix.lower().lstrip(".") or "png"
        mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
        data_uri = f"data:image/{mime};base64," + base64.b64encode(raw).decode("ascii")
        profile = self._str("edit_model", "qwen_edit").strip() or "qwen_edit"

        try:
            outcome = await self._generate(
                event,
                prompt,
                negative=negative_prompt,
                model=profile,
                image=data_uri,
                aspect=aspect,
            )
        except ImageApiError as exc:
            logger.warning("[imggen] 图生图失败：%s", exc)
            reason = str(exc).strip().rstrip("。.")
            return f"图生图失败：{reason}。可以直接告诉用户这次没改成。"
        except Exception as exc:  # pragma: no cover - 防御性兜底
            logger.exception("[imggen] 图生图异常")
            return f"图生图异常：{exc}"

        try:
            await event.send(event.chain_result(self._image_components(outcome)))
        except Exception:  # pragma: no cover
            logger.exception("[imggen] 发送图片失败")
            return "图片已生成但发送失败了，告诉用户稍后再试。"
        return (
            f"已改好并发送图片（{outcome.result.width}x{outcome.result.height}，"
            f"耗时 {outcome.result.duration:.1f}s）。不要重复调用工具，用一两句话自然衔接即可。"
        )

    # ------------------------------------------------------ 放大：SeedVR2 超分
    @filter.llm_tool(name="upscale_image")
    async def upscale_image_tool(
        self,
        event: AstrMessageEvent,
        image_id: int = 0,
        scale: float = 0,
    ) -> str:
        """把图片放大（超分）。用户说「放大」「超分」「高清一点」「4K」时用。

        Args:
            image_id(int): 可选，消息里 [图#12] 的编号；不传就用最近一张（当前消息/引用/刚出的图）
            scale(float): 可选，放大倍数，默认用配置（4 倍）；一般填 2 或 4
        """
        if not self._bool("enable_tool", True):
            return "生图功能当前已关闭，直接回答用户即可。"
        if not self._bool("upscale_enabled", True):
            return "超分功能当前已关闭。"
        umo = getattr(event, "unified_msg_origin", "") or ""
        refs = []
        if image_id:
            picked = _lazy_ref_path(umo, image_id)
            refs = [picked] if picked else []
        if not refs:
            refs = await self._collect_reference_images(event)
        if not refs:
            latest = _lazy_latest_path(umo, str(getattr(event, "get_sender_id", lambda: "")() or ""))
            if latest:
                refs = [latest]
        if not refs:
            return "没找到要放大的图：让用户先发一张，或者引用那张图再试。"

        src = Path(refs[0])
        try:
            raw = src.read_bytes()
        except OSError as exc:
            return f"读不到图片：{exc}"
        suffix = src.suffix.lower().lstrip(".") or "png"
        mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
        data_uri = f"data:image/{mime};base64," + base64.b64encode(raw).decode("ascii")

        factor = float(scale) if scale else float(self._int("upscale_scale", 4) or 4)
        factor = max(1.0, min(8.0, factor))
        tile = self._int("upscale_tile", 512) or 512
        overlap = self._int("upscale_overlap", 128) or 128
        color = self._str("upscale_color", "lab") or "lab"
        try:
            result = await asyncio.to_thread(self._client().upscale, data_uri, factor, tile, overlap, color)
        except ImageApiError as exc:
            logger.warning("[imggen] 超分失败：%s", exc)
            return f"放大失败：{src.name} → {exc}"
        except Exception as exc:  # pragma: no cover
            logger.exception("[imggen] 超分异常")
            return f"放大异常：{exc}"

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.image_dir / f"{stamp}_upscale.png"
        out_path.write_bytes(result.image)
        try:
            _lazy_register(umo, str(out_path), result.width, result.height,
                           str(getattr(event, "get_sender_id", lambda: "")() or ""))
        except Exception:
            logger.debug("[imggen] 超分结果登记失败", exc_info=True)

        try:
            await event.send(event.chain_result([Image.fromFileSystem(str(out_path))]))
        except Exception:  # pragma: no cover
            logger.exception("[imggen] 发送放大图失败")
            return "图放大了但发送失败，告诉用户稍后再试。"
        return (
            f"已放大并发送：{result.width}x{result.height}（{factor:g} 倍，耗时 {result.duration:.1f}s）。"
            "不要重复调用，用一两句话衔接即可。"
        )

    # ------------------------------------------------------ 反推：图 → 提示词
    async def _collect_reference_images(self, event: AstrMessageEvent) -> list[str]:
        """当前消息里的图优先，没有就找引用消息里的图。"""
        refs: list[str] = []
        message_obj = getattr(event, "message_obj", None)
        for comp in getattr(message_obj, "message", []) or []:
            if isinstance(comp, Image):
                try:
                    refs.append(await comp.convert_to_file_path())
                except Exception:  # pragma: no cover - 图片过期/下载失败
                    logger.debug("[imggen] 当前消息图片取不到", exc_info=True)
        if refs:
            return refs
        try:
            from astrbot.core.utils.quoted_message_parser import (
                extract_quoted_message_images,
            )

            refs = await extract_quoted_message_images(event)
        except Exception:
            logger.debug("[imggen] 引用图片解析失败", exc_info=True)
        return [r for r in (refs or []) if r]

    def _reverse_provider(self, event: AstrMessageEvent):
        """反推用的视觉模型：插件配置 > provider_settings 里的 caption provider > 当前对话模型。"""
        pid = self._str("reverse_prompt_provider", "").strip()
        if pid:
            prov = self.context.get_provider_by_id(pid)
            if prov is not None:
                return prov
        try:
            settings = (
                self.context.get_config(umo=event.unified_msg_origin) or {}
            ).get("provider_settings", {}) or {}
        except Exception:
            settings = {}
        cid = settings.get("default_image_caption_provider_id") or ""
        if cid:
            prov = self.context.get_provider_by_id(cid)
            if prov is not None:
                return prov
        try:
            return self.context.get_using_provider(umo=event.unified_msg_origin)
        except Exception:
            return None

    @filter.llm_tool(name="describe_image_as_prompt")
    async def describe_image_as_prompt_tool(
        self,
        event: AstrMessageEvent,
        hint: str | None = None,
        image_id: int = 0,
    ) -> str:
        """把用户刚发的图片（或引用的图片）反推成一段可直接生图的专业提示词。

        只有用户明确要求「照着这张图 / 上一条图来画」时才调用；普通文字描述画图不要调。

        Args:
            hint(string): 可选，用户对这张图的额外要求，比如"只要人物""换成夜景"
            image_id(int): 可选，消息里 [图#12] 的编号；指旧图时用它，不传就用当前消息/引用的图
        """
        refs = []
        if image_id:
            picked = _lazy_ref_path(getattr(event, "unified_msg_origin", "") or "", image_id)
            refs = [picked] if picked else []
        if not refs:
            refs = await self._collect_reference_images(event)
        if not refs:
            return "没找到可反推的图片：让用户先发一张图，或者引用那张图再试。"

        model = self._current_model(event)
        lang = "英文" if model == "krea2" else "中英混合（中文元素保留中文原词）"
        instruction = REVERSE_INSTRUCTION.format(lang=lang)
        if hint and hint.strip():
            instruction += f"\n\n额外要求：{hint.strip()}"

        prov = self._reverse_provider(event)
        if prov is None:
            return "反推失败：没有可用的视觉模型（检查 default_image_caption_provider_id 或插件配置 reverse_prompt_provider）。"

        try:
            resp = await prov.text_chat(prompt=instruction, image_urls=refs[:1])
            text = (getattr(resp, "completion_text", "") or "").strip()
        except Exception as exc:
            logger.warning("[imggen] 反推提示词失败：%s", exc)
            return f"反推失败：{exc}"
        if not text:
            return "反推失败：视觉模型没返回内容。"
        logger.info("[imggen] 反推提示词：%s", text[:150])
        return f"反推出的提示词（直接作为 generate_image 的 prompt 传进去）：\n{text}"

    def _spec_from_config(self, key: str, fallback: str) -> str:
        """读提示词规则：留空用内置；@文件名 从插件目录读整份模板；否则当正文。"""
        raw = self._str(key, "").strip()
        if not raw:
            return fallback
        if raw.startswith("@"):
            path = Path(raw[1:].strip())
            if not path.is_absolute():
                path = Path(__file__).parent / path
            try:
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return text
            except OSError:
                logger.warning("[imggen] 读不到提示词模板 %s，用内置", path)
            return fallback
        return raw

    # -------------------------------------------------------------- 提示词注入
    @filter.on_llm_request()
    async def inject_draw_hint(self, event: AstrMessageEvent, req) -> None:
        """每次 LLM 请求前把使用说明塞进 system_prompt。"""
        if not self._bool("inject_prompt", True):
            return
        if not self._bool("enable_tool", True):
            return
        if not self._session_allowed(event):
            return
        custom = self._str("inject_text", "").strip()
        if custom:
            text = custom
        else:
            # 按当前会话的模型，只注入那一条语言规则（不混着说）
            model = self._current_model(event)
            side = self._current_long_side(event)
            label = MODEL_LABELS.get(model) or (model or "后端默认")
            rule = LANG_RULE
            t2i = self._spec_from_config("prompt_t2i", T2I_SPEC.format(lang=rule))
            edit = self._spec_from_config("prompt_edit", EDIT_SPEC.format(lang=rule))
            text = "\n\n".join([INJECT_COMMON, t2i, edit])
            text += f"\n当前模型：{label}，长边 {side}"
        marker_on = self._bool("enable_marker", False)
        marker = self._str("marker_token", "[DRAW]").strip() or "[DRAW]"
        if marker_on and not custom:
            text += (
                f"\n- 如果 `generate_image` 工具不可用：在回复的最后**另起一行**输出 "
                f"`{marker} 英文提示词`，系统会把它替换成图片。"
            )
        try:
            req.system_prompt = (getattr(req, "system_prompt", "") or "") + "\n\n" + text
        except Exception:  # pragma: no cover
            logger.debug("[imggen] 注入系统提示失败", exc_info=True)

    # -------------------------------------------------------- 备用：标记触发
    @filter.on_decorating_result()
    async def handle_marker(self, event: AstrMessageEvent) -> None:
        """模型不支持工具调用时，把回复里的 `[DRAW] xxx` 换成真图。"""
        if not self._bool("enable_marker", False):
            return
        if not self._session_allowed(event):
            return
        try:
            result = event.get_result()
        except Exception:  # pragma: no cover
            return
        if not result or not getattr(result, "chain", None):
            return
        marker = self._str("marker_token", "[DRAW]").strip() or "[DRAW]"
        prompts: list[str] = []
        kept = []
        for comp in result.chain:
            if isinstance(comp, Plain) and marker in (comp.text or ""):
                cleaned, found = parse_marker(comp.text, marker)
                prompts.extend(found)
                if cleaned.strip():
                    comp.text = cleaned
                    kept.append(comp)
                continue
            kept.append(comp)
        if not prompts:
            return
        result.chain = kept
        try:
            outcome = await self._generate(event, prompts[0])
            result.chain.extend(self._image_components(outcome))
        except Exception as exc:
            logger.warning("[imggen] 标记出图失败：%s", exc)
            result.chain.append(Plain(f"（想配张图的，但没画成：{exc}）"))

    # ------------------------------------------------------------------ 命令
    @filter.command("draw", alias={"画图", "画画"})
    async def cmd_draw(self, event: AstrMessageEvent, prompt: GreedyStr):
        """手动出图：/draw 一只在雪地里的白狐"""
        text = str(prompt or "").strip()
        if not text:
            yield event.plain_result("用法：/draw 一只在雪地里的白狐，胶片质感")
            return
        event.stop_event()
        try:
            outcome = await self._generate(event, text)
        except ImageApiError as exc:
            yield event.plain_result(f"生图失败：{exc}")
            return
        except Exception as exc:  # pragma: no cover
            logger.exception("[imggen] 命令出图异常")
            yield event.plain_result(f"生图异常：{exc}")
            return
        chain = self._image_components(outcome)
        yield event.chain_result(chain)

    @filter.command("draw_model", alias={"画图模型"})
    async def cmd_model(self, event: AstrMessageEvent, name: GreedyStr):
        """查看或切换出图模型：/draw_model [zimage|krea2|qwen_image|default]"""
        event.stop_event()
        arg = str(name or "").strip().lower()
        try:
            info = await asyncio.to_thread(self._client().profiles)
        except Exception:
            info = {}
        profs = info.get("profiles", []) or []
        ids = [p.get("id") for p in profs if p.get("id")]
        labels = {p.get("id"): p.get("label", p.get("id")) for p in profs}
        alias = {
            "qwen": "qwen_image", "qwen2.1": "qwen_image", "qwen21": "qwen_image",
            "qwen_image_2.1": "qwen_image", "z-image": "zimage", "z_image": "zimage",
            "zimage_turbo": "zimage", "krea": "krea2",
        }

        if not arg:
            cur = self._current_model(event)
            default_id = info.get("default") or ""
            current = cur or default_id
            lines = [f"当前：{labels.get(current, current) or '后端默认'}", "可选："]
            for p in profs:
                pid = p.get("id")
                lab = p.get("label")
                # label 和 id 一样时就别写成 "moss — moss"
                lines.append(f"· {pid}" if not lab or lab == pid else f"· {pid} — {lab}")
            yield event.plain_result("\n".join(lines))
            return

        target = alias.get(arg, arg)
        if target in ("default", "reset", "默认", "auto"):
            self._set_override(event, model=None)
            yield event.plain_result("已恢复默认模型。")
            return
        if ids and target not in ids:
            yield event.plain_result(f"没有这个模型：{arg}\n可选：{', '.join(ids)}")
            return
        self._set_override(event, model=target)
        yield event.plain_result(f"已切换到 {labels.get(target, target)}。")

    @filter.command("draw_size", alias={"画图尺寸"})
    async def cmd_size(self, event: AstrMessageEvent, size: GreedyStr):
        """查看或设置长边：/draw_size 1024 / 768（比例由模型按画面决定）"""
        event.stop_event()
        arg = str(size or "").strip().lower().replace("×", "x").replace("：", ":")
        cur = self._current_long_side(event)
        if not arg:
            yield event.plain_result(
                f"当前长边：{cur}（比例由模型按画面决定）\n可选：1024 · 768 · 1280 · default"
            )
            return
        if arg in ("default", "reset", "默认", "auto"):
            self._set_override(event, long_side=None)
            yield event.plain_result("已恢复默认长边。")
            return
        m = re.fullmatch(r"(\d+)(?:[x:](\d+))?", arg)
        if not m:
            yield event.plain_result(f"看不懂这个尺寸：{size}\n例子：1024、768（只设长边）")
            return
        side = max(int(m.group(1)), int(m.group(2) or 0))
        self._set_override(event, long_side=str(side), size=None)
        yield event.plain_result(f"已把长边切到 {side}，比例仍由模型决定。")

    @filter.command("draw_cache")
    async def cmd_cache(self, event: AstrMessageEvent):
        """清理 ComfyUI 缓存：卸载模型、删残留的临时参考图。"""
        event.stop_event()
        try:
            info = await asyncio.to_thread(self._client().clear_cache)
        except Exception as exc:
            yield event.plain_result(f"清理失败：{exc}")
            return
        info = info or {}
        unloaded = "是" if info.get("unloaded") else "否"
        before = info.get("vram_before", -1)
        after = info.get("vram_after", -1)
        vram = f"，显存 {before}→{after} MiB" if before >= 0 and after >= 0 else ""
        yield event.plain_result(
            f"已清理：卸载模型={unloaded}，"
            f"删临时参考图 {info.get('removed_files', 0)} 个（{info.get('freed_mb', 0)} MB）{vram}"
        )

    @filter.command("draw_status")
    async def cmd_status(self, event: AstrMessageEvent):
        """检查出图接口是否在线。"""
        event.stop_event()
        base = self._str("api_base", "")
        try:
            info = await asyncio.to_thread(self._client().health)
        except Exception as exc:
            yield event.plain_result(f"出图接口不可用：{exc}\n（{base}）")
            return
        stats = (info or {}).get("stats", {}) or {}
        system = stats.get("system", {}) or {}
        devices = stats.get("devices", []) or []
        vram = ""
        for dev in devices:
            if dev.get("vram_total"):
                vram = (
                    f"，显存 {round(dev.get('vram_free', 0) / 1073741824, 1)}"
                    f"/{round(dev.get('vram_total', 0) / 1073741824, 1)} GB 空闲"
                )
        yield event.plain_result(
            f"出图接口正常\n地址：{base}\n"
            f"引擎 {system.get('comfyui_version', '?')} / {system.get('os', '?')}{vram}"
        )
