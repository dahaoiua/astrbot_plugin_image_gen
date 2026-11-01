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
- negative_prompt 只放用户要求排除的元素。
- 不传尺寸；不改模型；出图后一两句话衔接，不复述提示词。"""

T2I_SPEC = """【文生图提示词】（Qwen-Image 官方写法）
用一段连贯英文写「正在看着成品图的人」的描述，不是关键词堆砌，也不是对模型下指令。

- 开头句（约 20 词）：The image is a [vertical/wide/square] [realistic/cinematic/minimalist/...] [photograph/poster/illustration/...] of [subject], [background and palette].
- 之后按序展开：medium → subject → action → relationships → composition → lighting → materials → exact lettering
- 用户明确要求的文字、数量、颜色、位置原样保留；没提到的背景、光线、氛围自己补合理值
- 画面里要出现的具体文字用引号括起来并交代位置，例如：a cream headline at the top reads exactly "TAKE THE SCENIC ROUTE"
- 比例属于 wh_ratio 字段，不要写进正文；也不要写分辨率
- **光线必须写到能被画出来的程度**，每张图至少交代三件事：
  1) 方位：from camera-left / side-lit / backlit / top light / window light from the right / bottom light
  2) 软硬与色温：hard shadows / soft diffused light；warm golden hour / cool blue hour / mixed neon
  3) 投影与明暗：long shadows stretching across the floor / deep shadows in the corners / high-key 或 low-key
  例：warm window light from camera-left, hard shadows across the bed, bright rim light on her hair, low-key background
- 禁止只写 "good lighting"、"cinematic lighting" 这类空话；有光就有影，明暗要成对比
- 每个属性紧跟它描述的对象，不要堆到句尾
- 要透明底时用固定句式：This is an RGBA image with transparency. [subject]. The image has alpha channel and the background is transparent.
- 语言：{lang}"""

EDIT_SPEC = """【图生图提示词】（Qwen-Image 官方模板）
第一句直接说改什么，动词点明对象与区域；保留项一句带过。直接套这个模板：
Edit. [具体改动指令]. Requested change: [instruction]. Preserve unrelated content and the image's framing. Keep the subject's identity and visual style consistent.

- 不要先描述成品画面；不要细写不需要改的部分
- 改光影时同样写具体，别只说 brighter：例如 relight as warm side light from camera-right, push the room into low-key, keep a rim light on her shoulder
- 多张参考图时按顺序引用 first image / second image
- 语言：{lang}"""



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

    def _default_size(self) -> tuple[int, int]:
        raw = self._str("default_size", "").strip()
        if raw:
            return parse_size(raw)
        return (
            self._int("default_width", 512) or 512,
            self._int("default_height", 512) or 512,
        )

    def _current_model(self, event: AstrMessageEvent) -> str:
        ov = self._overrides.get(self._session_key(event)) or {}
        return str(ov.get("model") or self._str("default_model", "")).strip().lower()

    def _current_size(self, event: AstrMessageEvent) -> tuple[int, int]:
        ov = self._overrides.get(self._session_key(event)) or {}
        raw = str(ov.get("size") or self._str("default_size", "")).strip()
        return parse_size(raw) if raw else self._default_size()

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

        if size:
            width, height = int(size[0]), int(size[1])
        elif not (width and height):
            width, height = self._current_size(event)
        width, height = int(width or 512), int(height or 512)
        if steps is None:
            configured = self._int("default_steps", 0)
            steps = configured if configured > 0 else None
        cfg_value = self._float("default_cfg", 1.0) if cfg is None else float(cfg)
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
                ),
                timeout=timeout + 15,
            )
        except asyncio.TimeoutError as exc:
            raise ImageApiError(f"出图超时（>{timeout}s）。") from exc

        self._mark_used(event)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.image_dir / f"{stamp}_{result.seed}.png"
        path.write_bytes(result.image)
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
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
    ) -> str:
        """根据文字描述用配置好的出图服务生成一张图片，并直接发给用户。

        Args:
            prompt(string): 分层写的提示词短语（主体→外貌→服装→动作→环境→光影→镜头→画质），逗号分隔；以英文为主（Krea2 也允许中文文字），关键特征可用 (词:1.2) 权重
            negative_prompt(string): 可选，需要避免出现在画面里的元素
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
            )
        except ImageApiError as exc:
            logger.warning("[imggen] 工具出图失败：%s", exc)
            reason = str(exc).strip().rstrip("。.")
            return f"生图失败：{reason}。可以直接告诉用户这次没画成。"
        except Exception as exc:  # pragma: no cover - 防御性兜底
            logger.exception("[imggen] 工具出图异常")
            return f"生图异常：{exc}"

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

    # ------------------------------------------------------ 图生图：改一张图
    @filter.llm_tool(name="edit_image")
    async def edit_image_tool(
        self,
        event: AstrMessageEvent,
        prompt: str,
        negative_prompt: str | None = None,
        image_id: int = 0,
    ) -> str:
        """按用户的话改一张图（图生图）：用户发图后说「改成…」「换成…」「加个…」时用。

        Args:
            prompt(string): 具体怎么改，例如"把背景换成夜晚的街道""把裙子改成红色""给她加一顶帽子"
            negative_prompt(string): 可选，不要出现的元素
            image_id(int): 可选，消息里 [图#12] 的编号；改旧图时用它，不传就用当前消息/引用的图
        """
        if not self._bool("enable_tool", True):
            return "生图功能当前已关闭，直接回答用户即可。"
        refs = []
        if image_id:
            picked = _lazy_ref_path(getattr(event, "unified_msg_origin", "") or "", image_id)
            refs = [picked] if picked else []
        if not refs:
            refs = await self._collect_reference_images(event)
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
            w, h = self._current_size(event)
            label = MODEL_LABELS.get(model) or (model or "后端默认")
            rule = LANG_RULE
            t2i = self._str("prompt_t2i", "").strip() or T2I_SPEC.format(lang=rule)
            edit = self._str("prompt_edit", "").strip() or EDIT_SPEC.format(lang=rule)
            text = "\n\n".join([INJECT_COMMON, t2i, edit])
            text += f"\n当前模型：{label}，{w}x{h}"
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
        """查看或切换出图分辨率：/draw_size 1024x1024 / 16:9 / 768"""
        event.stop_event()
        arg = str(size or "").strip().lower().replace("×", "x").replace("：", ":")
        cur_w, cur_h = self._current_size(event)
        if not arg:
            yield event.plain_result(
                f"当前尺寸：{cur_w}x{cur_h}\n"
                "可选：1024x1024 · 16:9 · 9:16 · 768 · default"
            )
            return
        if arg in ("default", "reset", "默认", "auto"):
            self._set_override(event, size=None)
            yield event.plain_result("已恢复默认尺寸。")
            return
        if not re.fullmatch(r"\d+(\.\d+)?([x:]\d+(\.\d+)?)?", arg):
            yield event.plain_result(f"看不懂这个尺寸：{size}\n例子：1024x1024、16:9、768")
            return
        w, h = parse_size(arg)
        self._set_override(event, size=f"{w}x{h}")
        yield event.plain_result(f"已切换到 {w}x{h}。")

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
