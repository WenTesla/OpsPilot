"""接口连通性验证：上传 → 列表 → SSE 对话（需先启动 uvicorn）。"""
import json
import urllib.request

BASE = "http://127.0.0.1:8000"


def post(path, data=None, files=None):
    if files:
        import uuid
        boundary = uuid.uuid4().hex
        body = b""
        for k, v in data.items():
            body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
        fn, content = files
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                 f"filename=\"{fn}\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode()
        body += content + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(BASE + path, data=body, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    else:
        req = urllib.request.Request(BASE + path, data=json.dumps(data or {}).encode(), method="POST")
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read().decode()


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return r.status, r.read().decode()


MD = """# SOP-主机资源异常处理

## 3. CPU 使用率过高
### 3.1 快速定位
确认告警范围是单实例还是多实例，执行 top -c 按 CPU 排序定位进程。
### 3.2 常见根因
业务流量突增、死循环或正则回溯、JVM GC 频繁。先摘流量止损，再保留现场排查。

## 5. 磁盘空间不足
使用 df -h 确认挂载点，du -xh --max-depth=2 /data 定位大目录。
可安全清理轮转日志、/tmp 临时文件、core dump；禁止删除未备份的业务数据。
"""

print("health:", get("/api/health")[1])
print("upload:", post("/api/documents", {"doc_type": "sop", "service": "order-service", "env": "prod"},
                      files=("SOP-主机资源异常处理.md", MD.encode("utf-8"))))
print("list:", get("/api/documents")[1][:300])

status, body = post("/api/chat", {
    "conversation_id": "api-test-1",
    "messages": [{"role": "user", "content": "CPU 使用率超过 90% 怎么排查？"}],
})
print("chat status:", status)
print(body[:1500])
