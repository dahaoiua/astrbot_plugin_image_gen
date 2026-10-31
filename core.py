# -*- coding: utf-8 -*-
"""与 AstrBot 解耦的核心逻辑：提示词拼装、标记解析、出图 API 客户端。

后端是「任意实现了下面这套简单 HTTP 协议的服务」：

    GET <base>/healthz
    GET <base>/generate?prompt=...&negative_prompt=...&width=..&height=..&steps=..&cfg=..&seed=..
        -> 200 image/png（响应头带 X-Seed / X-Prompt-Id / X-Duration）

默认对接本机 8190 的 ComfyUI 包层，换成别的后端只要实现同样的协议即可。
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

# 默认画质词。英文 tag 在多数模型上更稳，调用方可以在插件配置里整段替换。
DEFAULT_POSITIVE = (
    "masterpiece, best quality, ultra detailed, sharp focus, "
    "cinematic lighting, beautiful composition, high dynamic range"
)
DEFAULT_NEGATIVE = (
    "lowres, blurry, jpeg artifacts, watermark, text, signature, logo, "
    "worst quality, low quality, deformed, bad anatomy, bad hands, extra fingers, cropped"
)


class ImageApiError(RuntimeError):
    """出图失败：连不上、上游报错、超时等。"""


@dataclass
class GenResult:
    """一次出图的结果。"""

    image: bytes
    seed: int
    prompt_id: str
    duration: float
    width: int
    height: int
    steps: int
    cfg: float
    content_type: str = "image/png"
    profile: str = ""


def _clean(text: str) -> str:
    """压掉换行和多余空白，去掉首尾的逗号/分号。"""
    text = (text or "").replace("\r", " ").replace("\n", " ")
    return " ".join(text.split()).strip(" ,，;；")


def compose_positive(
    user_prompt: str,
    quality: str = DEFAULT_POSITIVE,
    enabled: bool = True,
    max_chars: int = 1800,
) -> str:
    """用户/LLM 的提示词 + 内置画质词。画质词重复出现时不重复追加。"""
    prompt = _clean(user_prompt)
    if enabled:
        extra = _clean(quality)
        if extra and extra.lower() not in prompt.lower():
            prompt = f"{prompt}, {extra}" if prompt else extra
    if len(prompt) > max_chars:
        # 尽量在逗号处截断，别把最后一个词切断
        prompt = prompt[:max_chars]
        cut = prompt.rfind(",")
        if cut > max_chars * 0.6:
            prompt = prompt[:cut]
        prompt = prompt.strip(" ,")
    return prompt


def compose_negative(
    user_negative: str,
    quality: str = DEFAULT_NEGATIVE,
    enabled: bool = True,
) -> str:
    """内置负面词 + LLM 给的负面词，按 tag 去重。"""
    parts: list[str] = []
    seen: set[str] = set()
    for chunk in (quality if enabled else "", user_negative):
        for tag in _clean(chunk).split(","):
            tag = _clean(tag)
            if not tag or tag.lower() in seen:
                continue
            seen.add(tag.lower())
            parts.append(tag)
    return ", ".join(parts)


def parse_marker(text: str, marker: str = "[DRAW]") -> tuple[str, list[str]]:
    """摘出文本里的 `[DRAW] 提示词` 行。

    返回 (去掉标记后的文本, [提示词, ...])。标记所在行的其余内容当作提示词。
    """
    if not text or not marker or marker not in text:
        return text, []
    kept: list[str] = []
    prompts: list[str] = []
    for line in text.splitlines():
        if marker in line:
            rest = line.split(marker, 1)[1].strip(" \t:：-—,，[]()")
            if rest:
                prompts.append(rest)
            continue
        kept.append(line)
    cleaned = "\n".join(kept).strip()
    return cleaned, prompts


def parse_size(text: str, default: tuple[int, int] = (512, 512),
               max_side: int = 1024, align: int = 8) -> tuple[int, int]:
    """解析尺寸：`1024`、`1024x768`、`768*768`、`16:9`、`9:16` 都认。

    返回 (w, h)，会按 align 对齐并限制在 [256, max_side]。
    """
    raw = (text or "").strip().lower().replace("×", "x").replace("*", "x").replace("：", ":")
    if not raw:
        return default

    def _norm(v: int) -> int:
        v = max(256, min(max_side, int(v)))
        return max(256, (v // align) * align)

    try:
        if ":" in raw:
            a, b = raw.split(":", 1)
            a, b = float(a), float(b)
            if a <= 0 or b <= 0:
                return default
            if a >= b:
                w, h = max_side, int(round(max_side * b / a))
            else:
                w, h = int(round(max_side * a / b)), max_side
            return _norm(w), _norm(h)
        if "x" in raw:
            a, b = raw.split("x", 1)
            return _norm(int(float(a))), _norm(int(float(b)))
        side = _norm(int(float(raw)))
        return side, side
    except (ValueError, TypeError):
        return default


class ImageApiClient:
    """调用出图接口（默认走 SSH 反向隧道：http://127.0.0.1:8190）。"""

    def __init__(
        self,
        base_url: str,
        timeout: int = 300,
        user: str = "",
        password: str = "",
    ) -> None:
        self.base_url = (base_url or "").strip().rstrip("/")
        self.timeout = max(10, int(timeout or 300))
        self._auth = None
        if user:
            token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
            self._auth = f"Basic {token}"

    # ------------------------------------------------------------------ http
    def _open(self, url: str, accept: str = "application/json"):
        headers = {"Accept": accept}
        if self._auth:
            headers["Authorization"] = self._auth
        req = urllib.request.Request(url, headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:  # 4xx/5xx，带上游的错误信息
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:  # pragma: no cover
                pass
            raise ImageApiError(f"出图接口返回 {exc.code}：{detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise ImageApiError(f"连不上出图接口 {url}：{exc.reason}") from exc
        except TimeoutError as exc:
            raise ImageApiError("出图接口超时") from exc
        return resp

    def health(self) -> dict:
        """探活，返回 /healthz 的内容。"""
        resp = self._open(f"{self.base_url}/healthz")
        with resp:
            return json.loads(resp.read().decode("utf-8"))

    def profiles(self) -> dict:
        """取出图接口支持的模型列表（/profiles）。"""
        resp = self._open(f"{self.base_url}/profiles")
        with resp:
            return json.loads(resp.read().decode("utf-8"))

    def clear_cache(self) -> dict:
        """让后端清 ComfyUI 缓存：卸载模型 + 删残留的临时参考图（/cache/clear）。"""
        resp = self._post_json("/cache/clear", {}, accept="application/json")
        with resp:
            return json.loads(resp.read().decode("utf-8"))

    def generate(
        self,
        prompt: str,
        negative: str = "",
        width: int = 512,
        height: int = 512,
        steps: int | None = None,
        cfg: float = 1.0,
        seed: int | None = None,
        profile: str = "",
        image: str = "",
    ) -> GenResult:
        """出一张图，直接拿 PNG 字节（响应头里有 seed / prompt_id / duration / profile）。

        image 传入 data URI / base64 时会走 POST JSON（图生图），否则走原来的 GET。
        """
        params = {
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
            "cfg": float(cfg),
        }
        if steps:
            params["steps"] = int(steps)
        if profile:
            params["profile"] = profile
        if negative:
            params["negative_prompt"] = negative
        if seed is not None:
            params["seed"] = int(seed)
        if image:
            payload = dict(params)
            payload["image"] = image
            resp = self._post_json("/generate", payload, accept="image/png")
        else:
            url = f"{self.base_url}/generate?{urllib.parse.urlencode(params)}"
            resp = self._open(url, accept="image/png")
        with resp:
            body = resp.read()
            headers = resp.headers
            ctype = headers.get("Content-Type", "image/png")
        if not body.startswith(b"\x89PNG") and ctype.startswith("application/json"):
            # 正常情况下不会走到这里，兜底解析一下错误 JSON
            try:
                payload = json.loads(body.decode("utf-8"))
                raise ImageApiError(payload.get("error") or "出图接口没有返回图片")
            except json.JSONDecodeError as exc:
                raise ImageApiError("出图接口返回了无法解析的内容") from exc
        return GenResult(
            image=body,
            seed=int(headers.get("X-Seed") or 0),
            prompt_id=headers.get("X-Prompt-Id") or "",
            duration=float(headers.get("X-Duration") or 0.0),
            width=int(headers.get("X-Width") or width),
            height=int(headers.get("X-Height") or height),
            steps=int(headers.get("X-Steps") or steps or 0),
            cfg=float(cfg),
            profile=headers.get("X-Profile") or profile,
            content_type=ctype or "image/png",
        )

    def _post_json(self, path: str, payload: dict, accept: str = "application/json"):
        headers = {"Accept": accept, "Content-Type": "application/json"}
        if self._auth:
            headers["Authorization"] = self._auth
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:  # pragma: no cover
                pass
            raise ImageApiError(f"出图接口返回 {exc.code}：{detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise ImageApiError(f"连不上出图接口 {self.base_url}{path}：{exc.reason}") from exc
