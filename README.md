# ascend-sam3-llm-pipeline

基于昇腾 SAM3 和 LLM 的火焰 / 烟雾二次确认服务。

只接收前端上传的图片，不拉取视频流，也不控制前端采样周期。

处理流程：

1. POST /v1/frames 接收一张图片，入队后立即返回 202。
2. SAM3 一次检测 fire、smoke，不返回 mask。
3. 任一检测框分数严格大于 0.3 时，对同一张图片执行一次 LLM 检测。
4. LLM 明确确认火焰或烟雾时，保存原图、SAM3 标注图和 JSON 元数据。
5. 保存完成后调用报警占位函数；目前不会发送任何外部报警。
6. 未检出、不确定、模型请求失败、超时或回复解析失败，直接跳过图片，不重试、不重新采样。

没有任务 ID、任务查询接口、数据库、Redis、事件合并或告警冷却。
每张确认图片独立保存；同一视频流持续出现火焰时，会产生多份证据。

## 部署环境

与现有项目的服务器环境对齐：

| 项目 | 配置 |
|---|---|
| Docker Engine | 18.09 |
| docker-compose | 1.22，使用带连字符的旧命令 |
| Compose 文件格式 | 2.4，与 ascend-llm 一致 |
| 基础镜像 | 与 ascend-sam3 相同的 CANN 9.0.0 / Ubuntu 22.04 / Python 3.11 镜像 |
| 网络 | Linux host 网络 |
| NPU | 新服务不映射设备、不加载模型，不额外占用 NPU |
| 服务进程 | 一个 API 进程，内部异步调度 |
| 默认上传端口 | 18080 |
| 默认 SAM3 地址 | http://127.0.0.1:18000/predict/file |
| 默认 LLM 地址 | http://127.0.0.1:8080/v1 |

默认复用现有 SAM3 的基础镜像，避免为旧 Docker 引入未经验证的新运行环境。
虽然镜像含有 CANN，本服务只使用 Python、HTTP 和 Pillow，不访问驱动。

Dockerfile 不使用 BuildKit 的 RUN --mount、COPY --link 等功能。
Compose 不使用 profiles、宿主机 host-gateway、服务扩容或带默认值的变量插值。
必须先创建 .env。新的 Docker Compose 可能提示 version 已过时；
为了服务器上的 Compose 1.22 兼容性，请保留 version: "2.4"。

> 配置已按 Docker Compose 1.22.0 的官方 v2.4 schema 校验。
> 本地功能测试不依赖 NPU；它们不代表已完成 ARM64 容器构建或真实模型联调。

## 服务器启动

将整个项目上传到服务器，例如 /data/packages/ascend-sam3-llm-pipeline。
不要上传本机 .venv、.test-artifacts 和测试生成的 data。

~~~bash
cd /data/packages/ascend-sam3-llm-pipeline
cp -n .env.example .env
mkdir -p data/events
~~~

编辑 .env：

- 两个模型服务地址通常不需要修改。
- LLM_MODEL 留空时，第一次候选图片会读取 /v1/models；必须只返回一个模型。
  也可以填写当前服务的 served-model-name。
- 建议设置 PIPELINE_API_KEY，前端上传时携带 X-API-Key。
- 如果从浏览器跨域上传，填写 CORS_ORIGINS，包含协议、IP/域名和端口，
  三台机器对应的多个 origin 用逗号分隔。无需保证错峰。
- 图片存储路径由 PIPELINE_STORAGE_DIR 指定，必须能被容器用户写入。
  Compose 将其挂载为容器内的 /data/events。
- 无法访问 PyPI 时，可把 PIP_INDEX_URL 改成你信任的镜像地址。

先确认已有两个模型服务可访问：

~~~bash
curl -fsS http://127.0.0.1:18000/health
curl -fsS http://127.0.0.1:8080/v1/models
~~~

构建和启动本服务，不会重启或修改已有 SAM3、LLM 容器：

~~~bash
docker-compose config
DOCKER_BUILDKIT=0 docker-compose build
docker-compose up -d --no-build
docker-compose ps
curl -fsS http://127.0.0.1:18080/health
docker-compose logs -f --tail 100
~~~

docker-compose config 可能包含环境变量中的密钥，不要公开分享完整输出。

host 网络下不配置 ports 映射：PIPELINE_PORT 就是服务器监听端口。
只允许三台前端或可信内网访问该端口；如有防火墙，由运维按来源放行，
无需关闭整个防火墙。不要把模型端口或无鉴权上传服务暴露到公网。

停止服务：

~~~bash
docker-compose down
~~~

停止时最多等待 SHUTDOWN_TIMEOUT_SECONDS 排空任务，随后放弃尚未完成的图片。
已保存的证据位于宿主机挂载目录，不会因容器删除而丢失。

## 上传接口

### POST /v1/frames

multipart/form-data 字段：

| 字段 | 必填 | 说明 |
|---|---|---|
| image | 是 | JPEG、PNG 或单帧 WebP，默认最大 8 MiB、1600 万像素 |
| machine_id | 是 | 机器标识，1–64 个英文字母、数字、下划线或连字符 |
| stream_id | 是 | 视频流标识，同上；与 machine_id 组合唯一标识视频流 |
| stream_name | 否 | 展示名称，支持中文，最长 256 字符 |
| captured_at | 否 | 带时区的 ISO 8601 采集时间 |

~~~bash
curl -i http://SERVER_IP:18080/v1/frames \
  -H "X-API-Key: YOUR_KEY" \
  -F "image=@/data/test.jpg" \
  -F "machine_id=frontend-1" \
  -F "stream_id=camera-01" \
  -F "stream_name=仓库东门" \
  -F "captured_at=2026-08-28T10:00:00+08:00"
~~~

未设置 PIPELINE_API_KEY 时可以省略请求头。不要手动填写 multipart 的
Content-Type / boundary，由 curl、浏览器 FormData 或 HTTP 客户端生成。

接收成功，HTTP 202：

~~~json
{"accepted": true}
~~~

202 仅代表已经进入内存队列，不代表模型判断正常，也不保证最终会保存。
不返回任务 ID、不返回建议采样间隔、不提供 GET /v1/tasks 接口。

| 状态码 | 含义 |
|---|---|
| 202 | 入队成功 |
| 429 | SAM3 入口队列已满，该图片未接收 |
| 400 / 422 | 图片或参数无效 |
| 413 | 上传字节数或图片像素数超限 |
| 401 | 配置了上传密钥，但请求密钥错误 |
| 503 | 正在停止、工作协程异常或 HTTP 连接并发超限 |

前端按自己设定的周期继续上传，不需要等待模型处理、不要求补发图片。
服务不重新取帧、不安排复核、不假设前端能错峰。

### GET /health

仅检查本服务工作协程，并返回队列长度和当前进程累计计数：

~~~json
{
  "status": "ok",
  "upstreams": "not_checked",
  "queues": {"sam3": 0, "llm": 0},
  "counts": {"accepted": 10, "sam3_candidates": 2, "saved": 1, "llm_uncertain": 1}
}
~~~

不代表 SAM3 / LLM 已通过实际图片推理。计数在重启后归零，不是任务查询。
未发生的计数字段可能不出现。启动不要求模型在线；下游故障时相应图片直接跳过。

## 推理和队列

两个有界 FIFO 队列：

- SAM3_CONCURRENCY 个 SAM3 工作协程读取入口队列。
- 满足 fire/smoke、score > 0.3 的候选进入 LLM 队列，一张图只提交一次。
- LLM_CONCURRENCY 个工作协程独立完成 LLM 检测、证据保存和报警占位调用。
- LLM 队列满时，SAM3 工作协程等待候选入队，不丢弃已识别的候选。
  入口队列也满后，后续上传返回 429。
- SAM3 和 LLM 每张图片都只尝试一次；HTTP 客户端不开自动重试。
- 不合并同一路视频的图片，不保存普通阴性帧或不确定帧。

LLM 使用当前 ascend-llm 的 /chat/completions 接口，关闭思考模式，
temperature=0，输出限制默认 128 tokens。输出需要是严格 JSON：

~~~json
{"result": "fire", "reason": "可见橙色火焰"}
~~~

result 只允许 fire、smoke、fire_smoke、none、uncertain。
前三种保存；后两种跳过。无效 JSON、未知枚举、截断回复、请求失败、超时都跳过。
为兼容现有 Ascend 镜像，不强制依赖额外的结构化解码插件。
当前模型名会缓存，切换上游模型后需重启本服务或明确更新 LLM_MODEL。

重要运行限制：

- 必须使用一个 API 进程。不要增加 Uvicorn workers 或复制多个服务实例，
  否则全局队列和并发上限会被放大。
- 队列吸收突发，不能解决长期过载。队列太大会使旧图片等待很久。
- 内存队列不持久化，重启、崩溃或关机超时都会丢失未完成图片。
- 已有 SAM3 某些底层失败可能以空 results 返回；本服务无法仅凭该响应
  区分故障与未检出。这里没有修改 ascend-sam3 的错误协议。
- LLM 不确定或失败直接跳过，可能漏报；这是当前明确选择的处理方式。

## 保存格式

~~~text
data/events/
  2026-08-28/
    machine_frontend-1/
      stream_camera-01/
        020000_123456_随机标识/
          original.jpg
          annotated.jpg
          metadata.json
~~~

原图文件扩展名按实际格式确定，不依赖上传文件名。

- original：保留上传字节，不覆盖、不画框。
- annotated.jpg：在原图副本上绘制 SAM3 框、类别和分数；fire 为红色，
  smoke 为橙色。绘制所有通过 0.3 门槛的 SAM3 框，LLM 类别另记入 JSON。
- metadata.json：机器/视频流信息、采集/接收/确认时间、图片尺寸、SAM3 框和
  置信度、LLM 结论和原始回复、模型名、提示词版本以及两阶段调用耗时。
- 服务器时间使用 UTC 并带时区；前端提供的采集时间保留其时区。
- 对含 EXIF 旋转的图片，两个模型接收同一份方向归正的未标注图片；
  原始上传文件仍保持不变，标注图坐标按归正后的尺寸记录。
- 一组文件先写入临时目录，全部成功后改为正式目录；不会在保存失败后调用报警。
  硬中断可能留下 .tmp- 开头的不完整目录，它们不作为成功证据，也不自动恢复。
- 磁盘容量、保留天数由部署方管理；本版不自动删除证据。

## 报警对接位置

app/alarm.py 中的 send_alarm(event) 是唯一对接位置。
event 提供原图、标注图和 JSON 元数据路径，后续在这里实现上传协议。

当前函数只打印 alarm_not_configured，不发 HTTP 请求。
metadata.json 也明确记录 alarm.status=not_configured。
没有报警重试或去重；后续实现报警协议时需要同步定义投递状态的记录方式。

## 可调整参数

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| SAM3_CONCURRENCY | 4 | 访问 SAM3 的并发上限 |
| LLM_CONCURRENCY | 2 | LLM 阶段工作协程数量 |
| SAM3_QUEUE_SIZE | 15 | 等待 SAM3 的图片数上限 |
| LLM_QUEUE_SIZE | 15 | 等待 LLM 的候选数上限 |
| SAM3_TIMEOUT_SECONDS | 15 | 每张图片 SAM3 请求的总期限 |
| LLM_TIMEOUT_SECONDS | 60 | LLM 总期限，包含首次模型名查询 |
| LLM_MAX_TOKENS | 128 | LLM 输出长度上限 |
| SHUTDOWN_TIMEOUT_SECONDS | 30 | 停止时排空队列的最长等待 |
| MAX_IMAGE_BYTES | 8388608 | 单张图片上传字节上限 |
| MAX_IMAGE_PIXELS | 16000000 | 单张图片像素上限 |

上传完整 multipart 请求还允许额外 64 KiB 表单开销，超限请求会在表单解析前拒绝。
HTTP 连接并发上限为 64，图片解码和存储在线程中执行，不阻塞异步推理。

修改 .env 后重新创建本服务：

~~~bash
docker-compose up -d --no-build --force-recreate
~~~

只执行 restart 不会应用变更后的 Compose 环境变量。
前端调用间隔由你在真实服务器压测后决定，本服务不设置采样周期。

## 本地验证

Python 3.11 或更新版本，无需 CANN / NPU：

~~~bash
python -m venv .venv
# Linux:
. .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python scripts/smoke_test.py
~~~

单元/集成测试使用模拟的 HTTP 模型响应。
smoke_test.py 仅在 127.0.0.1 启动临时模型桩和真实 Uvicorn 进程，
并保存测试证据到 .test-artifacts；结束后关闭这两个测试服务。
它验证真实 HTTP 上传和文件保存，不验证模型精度、NPU 性能或 Docker 运行时。

参考：[Compose 1.22.0 v2.4 官方 schema](https://github.com/docker/compose/blob/1.22.0/compose/config/config_schema_v2.4.json)。
