# WhisperLiveKit 浏览器标签页音频扩展

这个扩展用于采集当前 Chromium 标签页的音频，并把音频送入 WhisperLiveKit 链路。

它适合：

- 直播回放字幕
- YouTube / Bilibili / 在线会议标签页转写
- 不方便走系统混音，但可以直接抓标签页音频的场景

> 当前版本只采集标签页音频，不会录制你的麦克风输入。

## 使用前提

你需要至少准备好下面其中一种后端：

- 直接启动的 `wlk`
- 或 `wlk-control-api` 管理下的 WLK + Bridge 链路

## 本地开发 / 安装

1. 从仓库根目录同步前端资源：

```bash
python scripts/sync_extension.py
```

2. 打开 Chrome 或其他 Chromium 浏览器
3. 进入扩展管理页
4. 开启「开发者模式」
5. 选择「加载已解压的扩展程序」
6. 选择当前的 `chrome-extension/` 目录

## 推荐配合方式

如果你在用控制平面，推荐顺序是：

1. 先启动 `wlk-control-api`
2. 确认 WLK Profile 已配置好模型与语言
3. 启动运行时
4. 再在浏览器里启用当前标签页音频采集

## 限制

- 只能抓当前标签页的音频
- 默认不抓麦克风
- Chromium 对 side panel / panel 模式下的音频采集限制很多

## 开发备注

- 扩展无法在 panel 模式下稳定采集标签页音频，相关讨论：
  - `https://issues.chromium.org/issues/40926394`
  - `https://groups.google.com/a/chromium.org/g/chromium-extensions/c/DET2SXCFnDg`
  - `https://issues.chromium.org/issues/40916430`
- 如果你一定要在扩展里接麦克风，可以参考这些技巧讨论：
  - `https://github.com/justinmann/sidepanel-audio-issue`
  - `https://medium.com/@lynchee.owo/how-to-enable-microphone-access-in-chrome-extensions-by-code-924295170080`
