# 🎀 彩虹朗读小屋｜每日英语单词 + 本地 AI 识别

**目标：** 每天由家长贴入课本中的 5–8 个英文单词或短句，孩子点击「听示范」，用麦克风跟读，由本机 Whisper AI 把发音转成文字并与目标词比对。**不使用 OpenAI API，不上传儿童声音，也没有按次计费。**

> 重要：这不是专业的发音测评工具。本地 Whisper 是语音转文字模型，短词识别可能误判。识别到目标词只能作为朗读练习参考，不能证明音素、重音、语调全部正确；没识别到也不代表发音错误。家长可手动确认。

## 启动

需要 Python 3.10+ 和电脑麦克风，建议在 Windows/macOS/Linux 电脑 Chrome 或 Edge 中使用。**Windows 用户可以直接双击 `start-windows.bat`**（先安装好 Python）；macOS/Linux 可以运行 `./start-mac-linux.sh`。如果启动失败，再按下面的命令手动操作。

```bash
# 在解压后的 daily-reading-buddy 文件夹内运行
python -m pip install -r requirements.txt
python server.py
```

打开 **http://localhost:8765**。

- 首次点「朗读」识别时，会自动下载 Whisper `tiny.en` 英语模型到当前电脑的 Hugging Face 模型缓存（大约数十 MB，需能访问下载源）；以后可离线使用。
- 模型在普通 CPU 上运行，首次加载及较慢机器上可能等待较久。
- 点击「⚙ 家长设置」，粘贴当日英语课本中的单词，一行一个，例如：

```text
apple | 苹果
blue | 蓝色
Good morning | 早上好
```

- 不需要自行购买 OpenAI API 额度，也不需要设置 API Key。
- 每天的单词和完成状态仅保存在当前浏览器的 localStorage，换浏览器或清除网站数据会丢失。录音只在临时文件中用于本机识别，识别完立刻删除。
- 服务只绑定到 `127.0.0.1`，手机或其他电脑默认无法访问，避免家庭局域网意外开放孩子声音上传入口。
- 浏览器语音示范使用设备内置语音引擎，其声音和效果因电脑而异。

## 家长可以怎样安排每天的练习

1. 把学校当天课文中要读的 5–8 个单词或短句贴进设置页面。
2. 让孩子听一遍，再念一遍。
3. AI 显示“听到了”就给一颗星；没有匹配就温柔地再试一次。
4. 如果 AI 对口音或者很短的词识别不准，由家长自行确认后通过。
5. 完成就结束，不要求反复刷到 100 分。

## 隐私、安全和未来的云端 AI

当前版本**完全不调用云端识别接口**，也没有将音频写入数据库。程序并不联网发送孩子的录音；首次下载模型时只获取模型文件。页面通过 localhost 提供，浏览器会请求麦克风权限。建议由家长启动和关闭。

将来如果想接 OpenAI 云端 API，尤其要注意孩子才 8 岁：OpenAI API 的未成年人指导要求在处理 13 岁以下儿童个人数据前实施 **Zero Data Retention (ZDR)**。不能仅凭购买 Plus 或开通普通 API Key 就直接上传孩子录音。必须先落实相应数据保留控制、合规要求，确认语音端点适用之后，再增设云端模式。官方文档：https://developers.openai.com/api/docs/guides/safety-checks/under-18-api-guidance

## 项目文件

本地音素评测引擎正在按 v2 方案开发，安装与命令行用法见 [引擎说明](speakstar_engine/README.md)。它提供独立的音素证据工具，尚未通过儿童准确率验收；当前页面仍使用上面的 Whisper 朗读流程。

```text
server.py              Python 本地 HTTP 与 Whisper 推理
requirements.txt       本地识别模型依赖
static/index.html      页面
static/style.css       粉彩游戏风格
static/app.js          每日词表、录音、听写比对、进度
README.md              使用说明
```

## 测试与局限

不接入真实孩子语音也能测试 HTTP 页面与词表解析；详见 `tests/`。真实语音推理需在用户电脑下载模型后验证，建议用不同背景噪声、英语口音、短词与句子测试。模型可能把单独的词识别成相似词，不能代替教师的专业发音反馈。
