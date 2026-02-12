from flask import Flask, request, render_template_string, redirect, url_for
from urllib.parse import quote
import os
import sqlite3
from pathlib import Path

from predict import predict_single_sql

"""
一个简单的 Web 登录演示，把你的 SQL-Transformer 模型当作 WAF：

流程：
- /login 页面提供 用户名 + 密码 输入框
- 提交时先用模型检测是否存在 SQL 注入风险
  - 如果判定为恶意：直接拦截，不再校验账号密码
  - 如果判定为正常：再校验账号密码是否匹配
"""

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "demo_users.db"


def init_db() -> None:
    """
    初始化一个简单的 SQLite 数据库，里面有一张 users 表：
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                password TEXT NOT NULL
            )
            """
        )
        # 确保至少有一个演示账号
        cur.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
        count = cur.fetchone()[0]
        if count == 0:
            cur.execute(
                "INSERT INTO users (username, password) VALUES ('admin', '123456')"
            )
        conn.commit()
    finally:
        conn.close()


def vulnerable_login(username: str, password: str) -> tuple[bool, str]:
    """
    演示用的“存在 SQL 注入风险”的验证方式：
    - 直接字符串拼接 SQL
    - 故意暴露数据库错误信息，用于演示报错注入
    - 不要在任何真实项目中这样写！

    在关闭 WAF 的前提下，你可以输入类似：
    - admin' OR 1=1-- （布尔盲注）
    - admin' AND 1=CAST((SELECT name FROM sqlite_master WHERE type='table') AS INTEGER)-- （报错注入）
    
    :return: (success: bool, error_msg: str) - 成功返回 (True, "")，失败返回 (False, error_message)
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        # 致命示范（故意不使用参数化查询）
        query = (
            "SELECT id, username FROM users "
            f"WHERE username = '{username}' AND password = '{password}'"
        )
        print(f"[DEBUG] Executing vulnerable query: {query}")
        cur.execute(query)
        row = cur.fetchone()
        return (row is not None, "")
    except sqlite3.Error as e:
        # 故意返回错误信息，用于演示报错注入
        error_msg = str(e)
        print(f"[DEBUG] SQL Error: {error_msg}")
        return (False, error_msg)
    finally:
        conn.close()

def env_flag(name: str, default: bool) -> bool:
    """
    读取环境变量开关：True/False, 1/0, yes/no, on/off（大小写不敏感）
    """
    import os

    v = os.environ.get(name)
    if v is None:
        return default
    v = str(v).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>SQL-Transformer WAF 登录演示</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif;
            background: #0f172a;
            color: #e5e7eb;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
        }
        .card {
            background: #111827;
            border-radius: 16px;
            padding: 32px 32px 24px;
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.9);
            width: 420px;
            box-sizing: border-box;
        }
        h1 {
            margin: 0 0 4px;
            font-size: 24px;
            font-weight: 600;
            color: #f9fafb;
        }
        .subtitle {
            margin-bottom: 20px;
            font-size: 13px;
            color: #9ca3af;
        }
        label {
            display: block;
            font-size: 13px;
            margin-bottom: 6px;
            color: #e5e7eb;
        }
        input[type="text"],
        input[type="password"] {
            width: 100%;
            padding: 9px 11px;
            border-radius: 8px;
            border: 1px solid #374151;
            background: #020617;
            color: #e5e7eb;
            font-size: 13px;
            box-sizing: border-box;
            outline: none;
            transition: border-color 0.15s, box-shadow 0.15s, background 0.15s;
        }
        input[type="text"]:focus,
        input[type="password"]:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 1px rgba(99,102,241,0.4);
            background: #020617;
        }
        .field {
            margin-bottom: 14px;
        }
        .btn-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 8px;
        }
        button {
            border: none;
            border-radius: 999px;
            padding: 9px 18px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.15s, box-shadow 0.15s, transform 0.05s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #4f46e5, #6366f1);
            color: white;
            box-shadow: 0 10px 25px rgba(79,70,229,0.55);
        }
        .btn-primary:hover {
            background: linear-gradient(135deg, #4338ca, #4f46e5);
            box-shadow: 0 14px 30px rgba(79,70,229,0.8);
            transform: translateY(-1px);
        }
        .btn-secondary {
            background: transparent;
            color: #9ca3af;
        }
        .result {
            margin-top: 18px;
            padding: 10px 12px;
            border-radius: 12px;
            font-size: 13px;
            line-height: 1.5;
        }
        .result-ok {
            background: rgba(22,163,74,0.1);
            border: 1px solid rgba(22,163,74,0.7);
            color: #bbf7d0;
        }
        .result-bad {
            background: rgba(220,38,38,0.07);
            border: 1px solid rgba(220,38,38,0.75);
            color: #fecaca;
        }
        .result-error {
            background: rgba(245,158,11,0.1);
            border: 1px solid rgba(245,158,11,0.7);
            color: #fde68a;
        }
        .result-title {
            font-weight: 600;
            margin-bottom: 4px;
        }
        .error-msg {
            margin-top: 8px;
            padding: 8px;
            background: rgba(0,0,0,0.3);
            border-radius: 6px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
            font-size: 11px;
            word-break: break-all;
            white-space: pre-wrap;
        }
        .meta {
            margin-top: 10px;
            font-size: 11px;
            color: #6b7280;
        }
        .meta code {
            background: #020617;
            padding: 2px 4px;
            border-radius: 4px;
            font-size: 11px;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>登录页</h1>
        <div class="subtitle"></div>

        <form method="post" action="/login">
            <div class="field">
                <label for="username">用户名</label>
                <input type="text" id="username" name="username" required
                       value="{{ username or '' }}" placeholder="admin">
            </div>
            <div class="field">
                <label for="password">密码</label>
                <input type="password" id="password" name="password" required
                       value="{{ password or '' }}" placeholder="123456">
            </div>
            <div class="field" style="display:flex; align-items:center; gap:8px; margin-top:-2px;">
                <input type="checkbox" id="enable_waf" name="enable_waf" {% if enable_waf %}checked{% endif %}
                       style="width:auto; transform: translateY(1px);">
                <label for="enable_waf" style="margin:0; cursor:pointer;">开启 WAF（模型检测）</label>
            </div>
            <div class="btn-row">
                <button type="submit" class="btn-primary">登录（触发 WAF）</button>
                <button type="button" class="btn-secondary"
                        onclick="document.getElementById('username').value=`admin' OR 1=1--`; document.getElementById('password').value=`anything`;">
                    填入典型注入 payload
                </button>
            </div>
        </form>

        {% if checked %}
            <div class="result {% if is_malicious %}result-bad{% elif db_error %}result-error{% elif auth_ok %}result-ok{% else %}result-bad{% endif %}">
                <div class="result-title">
                    {% if is_malicious %}
                        ⚠ 请求被 WAF 拦截：疑似 SQL 注入
                    {% elif db_error %}
                        ⚠ 数据库执行错误（报错注入信息）
                    {% elif auth_ok %}
                        ✅ WAF 放行 + 登录成功
                    {% else %}
                        ❌ WAF 放行，但账号或密码错误
                    {% endif %}
                </div>
                {% if not is_malicious %}
                    <div>模型判定为 SQL 注入的概率：<strong>{{ prob | round(4) }}</strong></div>
                {% endif %}
                {% if db_error %}
                    <div style="margin-top:8px; font-weight:600;">数据库错误信息：</div>
                    <div class="error-msg">{{ db_error }}</div>
                {% endif %}
            </div>

            <script>
                window.onload = function() {
                    {% if is_malicious %}
                        alert("⚠️ WAF 拦截：检测到疑似 SQL 注入！\\n\\n输入(截断): {{ display_text[:120] }}{% if display_text|length > 120 %}...{% endif %}");
                    {% else %}
                        {% if auth_ok %}
                            alert("✅ WAF 放行 + 登录成功！\\n\\n欢迎，{{ username }}");
                        {% else %}
                            alert("❌ WAF 放行，但账号或密码错误。");
                        {% endif %}
                    {% endif %}
                }
            </script>
        {% endif %}

        <div class="meta"></div>
    </div>
</body>
</html>
"""


def normalize_text(text: str) -> str:
    """对输入做简单清洗，并进行 URL 编码后，作为模型的输入文本。"""
    raw = (text or "").strip()
    if not raw:
        return ""
    return quote(raw, safe="")


@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        enable_waf = bool(request.form.get("enable_waf"))

        # 1）构造给模型看的检测文本（只包含用户输入，不包含字段名），编码交给 predict_single_sql 内部处理
        display_text = f"{username} {password}".strip()

        if not display_text:
            return render_template_string(
                HTML_TEMPLATE,
                checked=True,
                username=username,
                password=password,
                enable_waf=enable_waf,
                display_text="输入为空",
                combined_text="",
                is_malicious=False,
                prob=0.0,
                auth_ok=False,
                db_error="",
            )

        # 2）可选：先用你的模型做 WAF 检测（疑似注入则拦截）
        if enable_waf:
            pred_label, prob = predict_single_sql(display_text)
            is_malicious = bool(pred_label == 1)
        else:
            prob = 0.0
            is_malicious = False

        # 3）如果 WAF 放行，再做账号密码验证
        #    这里特意使用一个“有注入风险”的验证函数 vulnerable_login，方便你演示 SQL 注入：
        #    - 当 WAF 关闭时，可以用 payload 绕过验证
        #    - 当 WAF 开启并拦截时，请求不会进入数据库验证
        #    - 故意暴露数据库错误信息，用于演示报错注入
        db_error = ""
        if enable_waf and is_malicious:
            auth_ok = False
        else:
            auth_ok, db_error = vulnerable_login(username, password)

        return render_template_string(
            HTML_TEMPLATE,
            checked=True,
            username=username,
            password=password,
            enable_waf=enable_waf,
            display_text=display_text,
            combined_text="",  # 统一由 predict_single_sql 自己打印编码后的文本
            is_malicious=is_malicious,
            prob=prob,
            auth_ok=auth_ok,
            db_error=db_error,
        )

    # GET：只展示空表单
    enable_waf_default = env_flag("ENABLE_WAF", True)
    return render_template_string(
        HTML_TEMPLATE,
        checked=False,
        username="",
        password="",
        enable_waf=enable_waf_default,
        display_text="",
        combined_text="",
        is_malicious=False,
        prob=0.0,
        auth_ok=False,
        db_error="",
    )


if __name__ == "__main__":
    # 先初始化演示数据库（创建 users 表并插入默认账号）
    init_db()

    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"

    print("=" * 50)
    print("SQL注入检测WAF演示系统（用户名+密码）")
    print("=" * 50)
    print(f"访问地址: http://0.0.0.0:{port}/login")
    print(f"调试模式: {debug}")
    print("按 Ctrl+C 停止服务器")
    print("=" * 50)

    app.run(host="0.0.0.0", port=port, debug=debug)
