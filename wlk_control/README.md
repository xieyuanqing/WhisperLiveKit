# WLK Control API

`wlk_control` 是 WhisperLiveKit 的本地控制平面。

它把原本需要手工维护的多段命令、Profile JSON、模型目录和运行状态，统一成一个本地 API。

## 它负责什么

- 管理 Profile 的增删改查与激活状态
- 启动、停止、重启 `wlk` 与 `bridge_worker`
- 启动前检查：端口、FFmpeg、模型、音频设备、CUDA 运行时
- 管理官方模型下载与外部模型登记
- 提供日志 SSE 和 Bridge 电平流
- 输出最终命令预览，方便确认实际参数

## 适合的链路

推荐链路：

`系统音频 / 虚拟声卡 -> WhisperLiveKit -> Bridge -> LiveCaptions-Translator`

如果你已经有单独的前端控制台，通常只需要把它指向：

`http://127.0.0.1:18700/api`

## 快速开始

在仓库根目录运行：

```bash
python -m wlk_control.api --host 127.0.0.1 --port 18700
```

或者直接使用已注册命令：

```bash
wlk-control-api --host 127.0.0.1 --port 18700
```

## 默认数据目录

默认会在仓库根目录下创建：

- `.wlk-control/profiles.json`
- `.wlk-control/models_registry.json`
- `.wlk-control/models/`

如果要改位置，可以设置：

```bash
WLK_CONTROL_HOME=D:\custom-wlk-home
```

## 当前默认 Profile

控制平面内置的默认 Profile 面向日语本地直播链路：

- `id=jp-loopback-default`
- `model=large-v3-turbo`
- `language=ja`
- `backend_policy=localagreement`
- `backend=faster-whisper`
- `min_chunk_size=0.1`
- `ffmpeg_format=dshow`
- `sample_rate=16000`
- `channels=1`
- `chunk_ms=100`

并额外注入这些 LocalAgreement 调优参数：

- `buffer_trimming_sec=4`
- `long-silence-reset-sec=1.5`
- `max-active-no-commit-sec=13`
- `condition-on-previous-text=false`
- `beams=1`
- `no-speech-threshold=0.9`
- `compression-ratio-threshold=2.025`
- `vac-min-silence-duration-ms=200`
- `no-commit-force-sec=1.84`

## API 一览

### Health

- `GET /api/health`

### Profiles

- `GET /api/profiles`
- `GET /api/profiles/{id}`
- `POST /api/profiles`
- `PUT /api/profiles/{id}`
- `DELETE /api/profiles/{id}`
- `POST /api/profiles/{id}/activate`

### Runtime

- `GET /api/runtime/status?includeHealth=true`
- `POST /api/runtime/preflight`
- `POST /api/runtime/audio-devices`
- `POST /api/runtime/command-preview`
- `POST /api/runtime/start`
- `POST /api/runtime/stop`
- `POST /api/runtime/restart`
- `GET /api/runtime/logs/stream`
- `GET /api/runtime/meter/stream`

### Models

- `GET /api/models/catalog`
- `POST /api/models/register-path`
- `POST /api/models/path-details`
- `POST /api/models/download`
- `GET /api/models/jobs`
- `GET /api/models/jobs/{jobId}`
- `DELETE /api/models/register/{modelId}`
- `DELETE /api/models/managed/{modelId}`

## 常见调用示例

### 运行前检查

```json
POST /api/runtime/preflight
{
  "profile_id": "jp-loopback-default"
}
```

### 检测音频设备

```json
POST /api/runtime/audio-devices
{
  "ffmpeg_path": "ffmpeg",
  "ffmpeg_format": "dshow",
  "audio_device": "default"
}
```

### 预览最终命令

```json
POST /api/runtime/command-preview
{
  "profile_id": "jp-loopback-default"
}
```

## 下游对接建议

如果你用的是 `LiveCaptions-Translator`：

- `ASR Source = Whisper Bridge`
- `Whisper Bridge URL = ws://127.0.0.1:8765/captions`

## 排障顺序

1. `/api/health` 是否正常
2. `runtime/preflight` 哪一项失败
3. `runtime/command-preview` 是否带上预期参数
4. 日志流里先报错的是 `wlk` 还是 `bridge`
5. Windows + Faster-Whisper 时，重点看 CUDA 运行时检查结果

## 注意事项

- `.wlk-control/` 里通常含有本机音频设备名和本地路径，默认不建议提交
- `runtime/audio-devices` 适合先探测，再把返回值写回 `bridge.audio_device`
- 控制平面会先拉起 WLK，再拉起 Bridge，并持续监控两个进程
