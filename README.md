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
