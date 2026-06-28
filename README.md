# 后端滚动更新系统 v2

## 启动

```bash
cd /home/icesnls/workplace/cwxDoucument/数据合集/backend
/home/icesnls/anaconda3/envs/cwxtf/bin/pip install -r requirements.txt
/home/icesnls/anaconda3/envs/cwxtf/bin/python -m app.main
```

当前 `cwxtf` 环境已具备模型依赖，但缺少 `flask`，首次运行前需要安装 `requirements.txt`。

## 预测接口

四个接口请求结构一致：

```text
POST /predict_bingxi
POST /predict_C5
POST /predict_gandian
POST /predict_qiya
```

请求体：

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

`operation_records` 必填，且必须包含目标模型需要的全部操作变量。`lab_records` 可为空；如果前端有新化验值，应随请求一起发送。

同时兼容 OPC 后端当前请求结构：

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

响应会保留模型后端字段，也会返回 OPC 后端硬依赖的 `time` 和 `result`。

## 文件布局

所有运行依赖都放在 `runtime/` 下，代码只使用相对路径：

```text
runtime/models/<target>/active
runtime/models/<target>/candidates
runtime/models/<target>/archive
runtime/data/operation/<target>
runtime/data/lab/<target>
runtime/predictions/<target>
runtime/evaluations/<target>
runtime/logs
```

## 默认周期

```text
bingxi: 10d
gandian: none
qiya: 10d
C5: none
```

滚动更新周期不从预测请求读取，统一在 `.env` 中维护：

```text
ROLLING_CYCLE_BINGXI=10d
ROLLING_CYCLE_QIYA=10d
ROLLING_CYCLE_C5=none
ROLLING_CYCLE_GANDIAN=none
OPC_STALE_THRESHOLD_DAYS=7
```

`runtime/registry/model_registry.json` 仍维护 active/candidate 模型目录、更新时间等运行状态。

## 滚动更新

不提供手动触发 API。内部任务可调用：

```python
from app.services.rolling_update_service import run_daily_check

run_daily_check()
```

滚动更新逻辑：

1. active model 始终负责线上预测。
2. candidate model 只做后台影子评估。
3. 周期结束后，若 candidate MAE 优于 active，则切换到 active。
4. 使用当前周期带标签样本，从当前 active 继续训练新的 candidate。
5. `none` 周期只记录预测和评估，不生成 candidate。

## 注意

第一版使用文件存储，不处理多进程并发写入。生产多实例部署时应增加文件锁或改为数据库。

冷启动时，如果 OPC 请求里的操作数据时间与当前时间相差超过 `OPC_STALE_THRESHOLD_DAYS`，这些旧操作数据仍会临时参与当前预测，但不会保存到滚动更新数据池；详情记录在 `runtime/logs/prediction_cold_start/<target>.jsonl`。

## 扫码化验值接入

`scanner/` 是独立模块，不混入预测服务。它的职责只有三件事：

1. 通过串口读取二维码。
2. 连续识别到相同内容后再确认写入。
3. 将确认后的化验值追加到 `runtime/data/lab/<target>/YYYY-QN.xlsx`。

扫码模块默认读取 `.env` 中的配置：

```text
SCANNER_PORT=COM3
SCANNER_BAUD=9600
SCANNER_BOOT_INTERVAL=3.0
SCANNER_BOOT_TIMEOUT=2.0
SCANNER_SCAN_INTERVAL=5.0
SCANNER_SCAN_TIMEOUT=2.0
SCANNER_CONFIRM_COUNT=3
SCANNER_LOG_DIR=runtime/logs/scanner
```

说明：

- `SCANNER_CONFIRM_COUNT` 表示同一二维码连续命中的次数阈值。
- 扫码日志写到 `runtime/logs/scanner/scanner.jsonl`。
- 化验值不写 `.xls`，统一写入 backend 现有的季度 `.xlsx` 历史文件。
- 写入逻辑复用 `app/services/storage_service.py::append_lab_records()`，按 `sample_time` 去重并保持时间升序。

二维码文本解析要求包含四组字段：

```text
bx_entry / bx_time
c5_entry / c5_time
pr_entry / pr_time
tp_entry / tp_time
```

其中四个采样时间必须一致，否则不会写入。

扫码模块入口：

```bash
/home/icesnls/anaconda3/envs/cwxtf/bin/python -m scanner.scanner
```
