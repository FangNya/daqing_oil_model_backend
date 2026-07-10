# 后端滚动更新系统

本目录是模型后端，负责四类目标的预测、历史数据存储、滚动更新和扫码化验值写入。

## 启动

推荐使用已验证的 Conda 环境，不要直接使用系统默认 `python3`：

```bash
cd /home/icesnls/workplace/cwxDoucument/数据合集/backend
/home/icesnls/anaconda3/envs/cwxtf/bin/pip install -r requirements.txt
/home/icesnls/anaconda3/envs/cwxtf/bin/python -m app.main
```

默认监听：

```text
0.0.0.0:5000
```

健康检查：

```bash
curl http://127.0.0.1:5000/health
```

使用 `python -m app.main` 启动时，会同时启动滚动更新后台调度线程。若改用 `gunicorn`、`flask run` 或其他 WSGI 入口，需要额外确认是否调用了 `start_rolling_update_scheduler()`。

## 当前线上模型

预测时永远读取 registry 中配置的 active 模型，不会在启动时扫描所有模型目录并自动挑选。

当前四类目标的 active 模型目录：

```text
runtime/models/bingxi/active    # 丙烯，Informer
runtime/models/C5/active        # C5，XGBoost
runtime/models/gandian/active   # 干点，Informer
runtime/models/qiya/active      # 蒸气压，XGBoost
```

注册表：

```text
runtime/registry/model_registry.json
```

部署时必须带上 `runtime/models` 和 `runtime/registry/model_registry.json`。如果只部署源码而漏掉 active 模型文件，接口会返回安全兜底值，但真实模型没有工作。

## 预测接口

四个接口请求结构一致：

```text
POST /predict_bingxi
POST /predict_C5
POST /predict_gandian
POST /predict_qiya
```

标准请求体：

```json
{
  "operation_records": [
    {
      "TimeStamp": "2026-01-31 07:58:00"
    }
  ],
  "lab_records": [
    {
      "sample_time": "2026-01-31 08:00:00",
      "value": 62.3
    }
  ]
}
```

`operation_records` 必填，且必须包含目标模型需要的全部操作变量。`lab_records` 可为空；如果前端或扫码模块有新化验值，应随请求或扫码写入流程进入历史数据。

兼容 OPC 后端当前请求结构：

```json
{
  "endTime": 1760000000,
  "produce": [
    {
      "TimeStamp": "2026-01-31 07:58:00"
    }
  ]
}
```

响应会保留模型后端字段，并返回 OPC 后端硬依赖的 `time` 和 `result`：

```json
{
  "target": "qiya",
  "prediction_time": "2026-01-31 08:00:00",
  "prediction": 63.6,
  "time": "2026-01-31 08:00:00",
  "result": "63.6",
  "status": "ok"
}
```

接口返回层会做安全数值兜底：如果模型计算失败、返回空值、返回非数值、`nan` 或 `inf`，接口仍返回 HTTP 200，并保证 `result` 是可转为有限浮点数的字符串。

兜底值选择顺序：

1. 最近一次有效预测值。
2. 最近一次有效实验室值。
3. 每个目标的静态保底值。

如果使用了兜底，响应中会包含：

```json
{
  "fallback": true,
  "fallback_reason": "RuntimeError",
  "fallback_source": "latest_prediction"
}
```

## 运行态文件

所有运行依赖都在 `runtime/` 下：

```text
runtime/models/<target>/active
runtime/models/<target>/candidates
runtime/models/<target>/archive
runtime/registry/model_registry.json
runtime/data/operation/<target>
runtime/data/lab/<target>
runtime/predictions/<target>
runtime/evaluations/<target>
runtime/logs
```

这些文件是运行态，不是临时缓存。生产部署、迁移和备份时不要删除 active 模型、registry、操作历史、实验室历史和预测历史。

## 配置

配置文件：

```text
.env
```

滚动更新周期：

```text
ROLLING_CYCLE_BINGXI=10d
ROLLING_CYCLE_QIYA=10d
ROLLING_CYCLE_C5=none
ROLLING_CYCLE_GANDIAN=none
```

有效周期只有：

```text
none
10d
20d
30d
```

`none` 表示该目标不生成 candidate，也不会自动替换 active 模型；仍可记录预测和评估。

旧数据阈值：

```text
OPC_STALE_THRESHOLD_DAYS=7
```

滚动更新后台调度可选配置：

```text
ROLLING_UPDATE_ENABLED=true
ROLLING_UPDATE_INITIAL_DELAY_SECONDS=60
ROLLING_UPDATE_INTERVAL_SECONDS=86400
```

修改 `.env` 后需要重启后端。

## 滚动更新

生产环境使用 `python -m app.main` 启动时，会自动启动后台调度线程：

1. 默认启动后延迟 60 秒执行第一次 `run_daily_check()`。
2. 之后默认每 86400 秒执行一次。
3. 每次检查四个目标：`bingxi`、`C5`、`gandian`、`qiya`。
4. 调度日志写入 `runtime/logs/rolling_update/scheduler.jsonl`。

滚动更新核心逻辑：

1. active model 始终负责线上预测。
2. 若 registry 中存在 `candidate_model_dir`，到期后会比较 candidate 与 active 在当前周期标签上的 MAE。
3. candidate MAE 更低时，candidate 才会被复制为 active。
4. 当前周期有足够化验标签时，会从当前 active 继续训练并生成新的 candidate。
5. `none` 周期不会生成 candidate。

滚动周期控制的是评估和增量训练使用的标签时间段，不是单次预测向前取数的窗口。单次预测窗口由各模型配置中的 `delay_hours`、`window_hours`、`max_points` 等参数决定。

内部仍可手动调用：

```python
from app.services.rolling_update_service import run_daily_check

run_daily_check()
```

生产只建议单实例运行。当前存储是本地文件和本地模型目录，没有跨进程文件锁；多实例同时写 `runtime` 会有风险。

## 冷启动和旧数据

如果 OPC 请求里的操作数据时间与当前时间相差超过 `OPC_STALE_THRESHOLD_DAYS`，系统会将其视为旧数据并尝试使用实验室历史值回退。详情记录在：

```text
runtime/logs/prediction_cold_start/<target>.jsonl
```

2026-07 新数据可以从当前 active 旧模型开始使用。后续只要持续写入新的操作数据和实验室标签，滚动更新调度会在周期到期后生成 candidate，并在效果更好时替换 active。

## 扫码化验值接入

`scanner/` 是独立模块，不混入预测服务。它的职责：

1. 通过串口读取二维码。
2. 连续识别到相同内容后再确认写入。
3. 将确认后的化验值追加到 `runtime/data/lab/<target>/YYYY-QN.xlsx`。

扫码配置同样读取 `.env`：

```text
SCANNER_PORT=COM3
SCANNER_BAUD=9600
SCANNER_BOOT_INTERVAL=3.0
SCANNER_BOOT_TIMEOUT=2.0
SCANNER_SCAN_INTERVAL=86400
SCANNER_SCAN_TIMEOUT=2.0
SCANNER_CONFIRM_INTERVAL=5.0
SCANNER_CONFIRM_COUNT=3
SCANNER_LOG_DIR=runtime/logs/scanner
```

二维码文本要求包含：

```text
bx_entry / bx_time
c5_entry / c5_time
pr_entry / pr_time
tp_entry / tp_time
```

四组采样时间按各自目标独立写入，不要求四个时间完全一致。

扫码模块入口：

```bash
cd /home/icesnls/workplace/cwxDoucument/数据合集/backend
/home/icesnls/anaconda3/envs/cwxtf/bin/python -m scanner.scanner
```

扫码日志：

```text
runtime/logs/scanner/scanner.jsonl
```

化验值统一写入季度 `.xlsx` 历史文件，并复用 `app/services/storage_service.py::append_lab_records()` 按 `sample_time` 去重。
