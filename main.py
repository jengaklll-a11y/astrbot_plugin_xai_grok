import asyncio
import json
import re
import sys
import time
import uuid
import io
import base64
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from urllib.parse import urljoin, urlparse

import httpx
import aiofiles
from astrbot.api import logger
from astrbot.api.all import *
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.api.message_components import Video, Image as AstrImage, Plain, Reply, At

# 必须引入 Pillow 进行裁剪和压缩
try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None
    logger.warning("未检测到 Pillow 库，图片处理功能不可用，建议安装: pip install Pillow")

class GrokMediaPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        
        # API配置
        self.server_url = config.get("server_url", "http://127.0.0.1:8000").rstrip('/')
        self.video_model_id = config.get("model_id", "grok-imagine-0.9")
        self.image_model_id = config.get("image_model_id", "grok-4.1-thinking")
        self.api_key = config.get("api_key", "")
        
        # 请求配置 (内嵌默认值)
        self.timeout_seconds = 180
        self.max_retry_attempts = 3
        
        # 5MB 阈值 (Base64膨胀后约6.6MB，避免触碰服务端10MB限制)
        self.max_image_size = 5 * 1024 * 1024 
        
        # 强制不保留文件，发送后自动清理
        self.save_video_enabled = False

        # 数据保存目录
        try:
            plugin_data_dir = Path(StarTools.get_data_dir("astrbot_plugin_xai_grok"))
            self.data_dir = plugin_data_dir / "downloads"
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.data_dir = self.data_dir.resolve()
        except Exception as e:
            logger.warning(f"无法使用StarTools数据目录: {e}")
            self.data_dir = Path(__file__).parent / "downloads"
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.data_dir = self.data_dir.resolve()
        
        self.api_url = urljoin(self.server_url + "/", "v1/chat/completions")
        logger.info(f"Grok多媒体插件已初始化，API: {self.api_url}")

    def _create_client(self, timeout: httpx.Timeout) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout)

    def _format_base64(self, base64_str: str) -> str:
        """仅格式化Base64字符串，添加前缀"""
        base64_str = base64_str.replace("\n", "").replace("\r", "")
        if not base64_str.startswith('data:'):
            return f"data:image/jpeg;base64,{base64_str}"
        return base64_str

    def _process_image_sync(self, base64_str: str, crop_for_video: bool = False) -> str:
        """
        同步图片处理逻辑（CPU密集型），应在 executor 中运行
        """
        if not PILImage:
            return self._format_base64(base64_str)

        try:
            # 1. 提取纯 Base64 数据
            if ',' in base64_str:
                header, data = base64_str.split(',', 1)
            else:
                data = base64_str
            
            # 2. 解码图片
            try:
                image_data = base64.b64decode(data)
            except Exception:
                data = re.sub(r'[^a-zA-Z0-9+/=]', '', data)
                image_data = base64.b64decode(data)

            # 3. 检查大小
            original_size = len(image_data)
            is_too_large = original_size > self.max_image_size

            # 如果既不需要裁剪，也不需要压缩，直接返回原图（最高画质）
            if not crop_for_video and not is_too_large:
                return self._format_base64(base64_str)

            # 4. 开始处理
            with io.BytesIO(image_data) as input_buffer:
                img = PILImage.open(input_buffer)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # --- 智能裁剪逻辑 (仅视频) ---
                if crop_for_video:
                    width, height = img.size
                    ratio = width / height
                    
                    # 判断目标比例
                    if 0.85 <= ratio <= 1.15:
                        # 接近正方形 -> 1:1
                        target_ratio = 1.0
                        logger_msg = "1:1 方形"
                    elif ratio > 1.15:
                        # 宽图 -> 16:9
                        target_ratio = 16 / 9
                        logger_msg = "16:9 横屏"
                    else:
                        # 竖图 -> 9:16
                        target_ratio = 9 / 16
                        logger_msg = "9:16 竖屏"

                    # 执行裁剪
                    if ratio > target_ratio:
                        # 图片比目标更宽，裁掉左右
                        new_width = int(height * target_ratio)
                        left = (width - new_width) // 2
                        img = img.crop((left, 0, left + new_width, height))
                    elif ratio < target_ratio:
                        # 图片比目标更高，裁掉上下
                        new_height = int(width / target_ratio)
                        top = (height - new_height) // 2
                        img = img.crop((0, top, width, top + new_height))
                    
                    logger.info(f"图片已自动裁剪为 {logger_msg}")

                # --- 压缩逻辑 (仅大图) ---
                save_kwargs = {"format": "JPEG"}
                if is_too_large:
                    # 限制最大分辨率，防止过大
                    img.thumbnail((2048, 2048), PILImage.Resampling.LANCZOS)
                    save_kwargs["quality"] = 80  # 稍微压缩
                    logger.info(f"图片过大({original_size/1024/1024:.2f}MB)，已压缩并调整尺寸")
                else:
                    # 保持极高画质
                    save_kwargs["quality"] = 95
                    save_kwargs["subsampling"] = 0

                # 5. 导出
                with io.BytesIO() as output_buffer:
                    img.save(output_buffer, **save_kwargs)
                    jpeg_data = output_buffer.getvalue()
                    new_base64 = base64.b64encode(jpeg_data).decode('utf-8')
                    return f"data:image/jpeg;base64,{new_base64}"
                    
        except Exception as e:
            logger.error(f"图片处理失败: {e}，将使用原图")
            return self._format_base64(base64_str)

    async def _extract_images_from_message(self, event: AstrMessageEvent, crop_for_video: bool = False) -> List[str]:
        images = []
        if hasattr(event, 'message_obj') and event.message_obj and hasattr(event.message_obj, 'message'):
            loop = asyncio.get_running_loop()
            for comp in event.message_obj.message:
                # 1. 处理直接上传的图片
                if isinstance(comp, Image):
                    try:
                        base64_data = await comp.convert_to_base64()
                        if base64_data:
                            # 优化：在线程池中运行CPU密集型的图片处理
                            processed_data = await loop.run_in_executor(
                                None, self._process_image_sync, base64_data, crop_for_video
                            )
                            images.append(processed_data)
                    except Exception: pass
                
                # 2. 处理回复中的图片
                elif isinstance(comp, Reply) and comp.chain:
                    for reply_comp in comp.chain:
                        if isinstance(reply_comp, Image):
                            try:
                                base64_data = await reply_comp.convert_to_base64()
                                if base64_data:
                                    # 优化：在线程池中运行
                                    processed_data = await loop.run_in_executor(
                                        None, self._process_image_sync, base64_data, crop_for_video
                                    )
                                    images.append(processed_data)
                            except Exception: pass
                
                # 3. 处理 @用户 (获取头像)
                elif isinstance(comp, At):
                    try:
                        target_qq = comp.qq
                        if target_qq:
                            logger.info(f"检测到@用户 {target_qq}，正在获取头像...")
                            avatar_url = f"https://q.qlogo.cn/headimg_dl?dst_uin={target_qq}&spec=640"
                            async with self._create_client(httpx.Timeout(30.0)) as client:
                                resp = await client.get(avatar_url)
                                if resp.status_code == 200:
                                    avatar_b64 = base64.b64encode(resp.content).decode('utf-8')
                                    # 优化：在线程池中运行
                                    processed_data = await loop.run_in_executor(
                                        None, self._process_image_sync, avatar_b64, crop_for_video
                                    )
                                    images.append(processed_data)
                                else:
                                    logger.warning(f"获取头像失败，状态码: {resp.status_code}")
                    except Exception as e:
                        logger.error(f"处理@用户头像异常: {e}")
                            
        return images

    async def _call_grok_api(self, prompt: str, image_base64: Optional[str], model: str) -> Tuple[Optional[str], Optional[str]]:
        if not self.api_key: return None, "未配置API密钥"
        
        content = [{"type": "text", "text": prompt}]
        if image_base64:
            content.append({"type": "image_url", "image_url": {"url": image_base64}})
            
        payload = {
            "model": model, 
            "messages": [{"role": "user", "content": content}]
        }
        
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        # 增加读取超时，视频生成可能很慢
        timeout_config = httpx.Timeout(connect=20.0, read=self.timeout_seconds, write=60.0, pool=self.timeout_seconds + 10)
        last_error = "未知错误"
        
        for attempt in range(self.max_retry_attempts):
            try:
                log_msg = f"调用Grok API (模型: {model}, 尝试 {attempt + 1}/{self.max_retry_attempts})"
                logger.info(log_msg)
                
                async with self._create_client(timeout_config) as client:
                    response = await client.post(self.api_url, json=payload, headers=headers)
                    logger.info(f"API响应: {response.status_code}")
                    
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            url, parse_error = self._extract_media_url_from_response(result)
                            if url: return url, None
                            
                            logger.error(f"提取媒体链接失败: {parse_error}")
                            last_error = parse_error or "未找到媒体链接"
                        except json.JSONDecodeError: last_error = "JSON解析失败"
                    
                    elif response.status_code == 429:
                        logger.warning("当前账号触发限流 (429)，等待 15s 后重试...")
                        last_error = "触发限流 (429)，正在重试..."
                        await asyncio.sleep(15) # 429 需要等待较长时间
                        continue

                    elif response.status_code == 403: 
                        return None, "API鉴权失败(403)，请检查API Key"
                    
                    elif response.status_code == 500:
                        error_text = response.text
                        logger.error(f"API 500 详情: {error_text}")
                        
                        # 解析 500 错误
                        if "429" in error_text:
                            last_error = "上游服务限流 (429)"
                        elif "void *" in error_text or "NoneType" in error_text:
                            last_error = "服务端浏览器实例崩溃，正在重试..."
                            # 这种错误通常需要时间恢复
                            await asyncio.sleep(10)
                            continue
                        elif "list index out of range" in error_text: 
                            last_error = "服务端处理失败"
                        else:
                            # 尝试解析 JSON 错误
                            try:
                                err_json = response.json()
                                if "error" in err_json:
                                    last_error = f"服务端错误: {err_json['error']}"
                                else:
                                    last_error = f"服务端错误(500): {error_text[:100]}"
                            except:
                                last_error = f"服务端错误(500): {error_text[:50]}"
                    else: 
                        last_error = f"API请求失败({response.status_code})"
                
                # 普通重试间隔
                if attempt < self.max_retry_attempts - 1:
                    wait_time = 5 * (attempt + 1)
                    logger.info(f"请求失败，{wait_time}秒后重试...")
                    await asyncio.sleep(wait_time)
            except Exception as e:
                last_error = f"请求异常: {str(e)}"
                if attempt < self.max_retry_attempts - 1: await asyncio.sleep(3)
        
        return None, last_error

    def _extract_media_url_from_response(self, response_data: dict) -> Tuple[Optional[str], Optional[str]]:
        try:
            if not isinstance(response_data, dict) or "choices" not in response_data: return None, "无效响应"
            choice = response_data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "")
            
            # 1. 检查字段
            if "video_url" in response_data: return response_data["video_url"], None
            if "image_url" in response_data: return response_data["image_url"], None
            if "video_url" in message: return message["video_url"], None
            
            if "render_searched_image" in content: return None, "Grok 执行了搜索而非生成，请尝试更具体的提示词。"
            
            # 2. Markdown 提取
            md_regex = r"!\[.*?\]\((https?://[^\s<>\"']+)\)"
            md_match = re.search(md_regex, content)
            if md_match: return md_match.group(1), None

            # 3. HTML 提取
            html_regex = r"""(?:src|href)=["'](https?://[^"']+)["']"""
            html_match = re.search(html_regex, content, re.IGNORECASE)
            if html_match: return html_match.group(1), None

            # 4. 暴力提取
            urls = re.findall(r"https?://[^\s<>\"')\]]+", content)
            trusted_domains = ["assets.grok.com", "assets.x.ai", "grok.com", "x.ai"]
            valid_exts = {".mp4", ".jpg", ".jpeg", ".png", ".webp", ".gif", ".mov", ".webm"}
            
            for url in urls:
                clean_url = url.rstrip(".,;:]}")
                try:
                    parsed = urlparse(clean_url)
                    path_lower = parsed.path.lower()
                    if any(path_lower.endswith(ext) for ext in valid_exts):
                        return clean_url, None
                    if any(d in parsed.netloc for d in trusted_domains) and len(parsed.path) > 1 and parsed.path != "/":
                        return clean_url, None
                except Exception: continue
            
            return None, "未提取到有效的媒体链接"
        except Exception as e: return None, f"提取异常: {e}"

    async def _download_file(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        下载文件，支持鉴权和自动重试
        返回: (local_path, error_msg_or_mime)
        """
        try:
            parsed = urlparse(url)
            path = parsed.path
            ext = Path(path).suffix.lower()
            if not ext: ext = ".mp4" 
            
            filename = f"grok_media_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}{ext}"
            file_path = self.data_dir / filename
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            }
            cookies = {}
            
            # --- 鉴权逻辑 ---
            is_self_hosted = False
            try:
                # 检查下载链接的主机是否与配置的API服务器主机一致
                server_netloc = urlparse(self.server_url).netloc
                if parsed.netloc == server_netloc:
                    is_self_hosted = True
            except: pass

            if is_self_hosted:
                # 自建服务：通常需要 API Key 才能访问静态资源
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                # 避免发送错误的 Referer
                headers["Referer"] = self.server_url
            elif "grok.com" in parsed.netloc:
                # 官方服务：需要 Cookie 和 Referer
                headers["Referer"] = "https://grok.com/"
                if self.api_key and len(self.api_key) > 50:
                     cookies = {"sso": self.api_key, "sso-rw": self.api_key}

            # --- 下载逻辑 ---
            async def do_download(target_url):
                async with self._create_client(httpx.Timeout(300.0)) as client:
                    response = await client.get(target_url, headers=headers, cookies=cookies)
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "")
                    
                    final_path = file_path
                    if content_type:
                        guess_ext = mimetypes.guess_extension(content_type.split(';')[0])
                        if guess_ext and guess_ext != ext and guess_ext not in ['.html', '.htm']: 
                            final_path = file_path.with_suffix(guess_ext)
                    
                    async with aiofiles.open(final_path, 'wb') as f:
                        await f.write(response.content)
                    return str(final_path.resolve()), content_type

            try:
                return await do_download(url)
            except Exception as e:
                # 失败回退：如果原始链接下载失败，且是自建服务，尝试强制使用配置的 server_url 拼接路径重试
                if is_self_hosted:
                    return None, f"下载失败({e})"
                
                # 如果不是自建服务，或者 URL 看起来不对，尝试回退
                try:
                    server_parsed = urlparse(self.server_url)
                    if parsed.netloc != server_parsed.netloc:
                        logger.warning(f"直接下载失败: {e}，尝试使用配置的 Server URL Host 重试...")
                        fallback_url = urljoin(self.server_url, path)
                        return await do_download(fallback_url)
                except: pass

                if isinstance(e, httpx.HTTPStatusError):
                    return None, f"HTTP {e.response.status_code}"
                return None, str(e)

        except Exception as e:
            logger.error(f"下载流程异常: {e}")
            return None, str(e)

    async def _cleanup_file(self, path: Optional[str]):
        # self.save_video_enabled 始终为 False，因此总是执行清理
        if not path or self.save_video_enabled: return
        try:
            p = Path(path)
            if p.exists(): p.unlink()
        except: pass

    async def _process_task(self, event: AstrMessageEvent, prompt: str, task_type: str, image_base64: Optional[str] = None):
        task_id = str(uuid.uuid4())[:8]
        
        # 1. 清理提示词中可能包含的"用户："前缀
        prompt = prompt.replace("用户：", "").replace("User:", "").strip()
        
        # 2. 清理移除标记后可能多余的空格
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        
        try:
            model = self.video_model_id if task_type == "video" else self.image_model_id
            action_name = {"video": "生成视频", "image": "生成图片", "edit": "修改图片"}.get(task_type, "任务")
            
            # 添加前缀以提示模型
            final_prompt = prompt
            if task_type == "image":
                final_prompt = f"Generate an image of {prompt}"
            elif task_type == "video":
                final_prompt = f"Animate this image: {prompt}"

            # 图标逻辑
            icon = "📺" if task_type == "video" else "🎨"
            
            yield event.plain_result(f"{icon} 正在{action_name}...")
            asyncio.create_task(self._async_core(event, final_prompt, image_base64, model, task_id))
        except Exception as e:
            logger.error(f"任务启动失败: {e}")
            yield event.plain_result(f"❌ 错误: {e}")

    async def _async_core(self, event: AstrMessageEvent, prompt: str, image_base64: Optional[str], model: str, task_id: str):
        local_path = None
        try:
            url, error = await self._call_grok_api(prompt, image_base64, model)
            
            if error:
                try:
                    chain = [Reply(id=str(event.message_obj.message_id)), Plain(f"❌ {error}")]
                    await event.send(event.chain_result(chain))
                except TypeError:
                    try:
                        chain = [Reply(), Plain(f"❌ {error}")]
                        await event.send(event.chain_result(chain))
                    except Exception as e_inner:
                        logger.warning(f"无法构建Reply组件: {e_inner}，降级为普通发送")
                        await event.send(event.plain_result(f"❌ {error}"))
                except Exception as e:
                    logger.error(f"发送错误提示失败: {e}")
                    await event.send(event.plain_result(f"❌ {error}"))
                return

            # download_file 现在返回 (path, mime_or_error)
            local_path, mime_or_err = await self._download_file(url)
            
            if not local_path:
                msg = "⚠️ 资源已生成，但下载失败。\n"
                if mime_or_err == "403" or mime_or_err == "HTTP 403": 
                    msg += "原因：403 Forbidden (无权访问链接，已尝试添加API Key但仍失败)。\n"
                elif mime_or_err:
                    msg += f"原因：{mime_or_err}\n"
                
                # 已删除：拼接原始链接的代码
                await event.send(event.plain_result(msg))
                return

            try:
                is_video = False
                ext = Path(local_path).suffix.lower()
                if ext in ['.mp4', '.mov', '.webm', '.avi', '.mkv']: is_video = True
                elif ext not in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp']: is_video = True
                
                if is_video:
                    component = Video.fromFileSystem(local_path)
                    await asyncio.wait_for(event.send(event.chain_result([component])), timeout=120.0)
                else:
                    component = AstrImage.fromFileSystem(local_path)
                    
                    # 构建带引用的消息链
                    chain = []
                    try:
                        chain.append(Reply(id=str(event.message_obj.message_id)))
                    except TypeError:
                        try: chain.append(Reply())
                        except: pass
                    except Exception: pass
                    
                    chain.append(component)
                    
                    await asyncio.wait_for(
                        event.send(event.chain_result(chain)),
                        timeout=120.0
                    )
            except asyncio.TimeoutError:
                await event.send(event.plain_result("⚠️ 发送超时，可能仍在传输中"))
            
        except Exception as e:
            logger.error(f"异步任务异常: {e}")
            await event.send(event.plain_result(f"❌ 异常: {e}"))
        finally:
            await self._cleanup_file(local_path)

    @filter.command("视频")
    async def cmd_video(self, event: AstrMessageEvent, *, prompt: str):
        """/视频 <提示词> (需附带图片)"""
        # 启用裁剪
        images = await self._extract_images_from_message(event, crop_for_video=True)
        if not images: yield event.plain_result("❌ 视频生成需要提供图片"); return
        async for res in self._process_task(event, prompt, "video", images[0]): yield res

    @filter.command("画图")
    async def cmd_image_gen(self, event: AstrMessageEvent, *, prompt: str):
        """/画图 <提示词> (附图则为图生图，纯文字为文生图)"""
        # 不启用裁剪
        images = await self._extract_images_from_message(event, crop_for_video=False)
        if images:
            # 有图 -> 图生图 (edit)
            async for res in self._process_task(event, prompt, "edit", images[0]): yield res
        else:
            # 无图 -> 文生图 (image)
            async for res in self._process_task(event, prompt, "image", None): yield res

