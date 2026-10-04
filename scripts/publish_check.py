#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""作品集发布前检查（脱敏 + 侵权风险提示）。

用法（在项目根目录执行）：
    python scripts/publish_check.py            # 只检查并给建议，不改动任何文件
    python scripts/publish_check.py --stash    # 把命中风险的文档移到 data/_quarantine/（可逆）

它做四件事：
  1. 检查 .env / 密钥是否可能被提交（git 已跟踪的文件里是否含 .env、sk- 开头的串）；
  2. 扫描 data/documents 下的文档，按来源标记出**公开传播风险较高**的文件；
  3. 检查运行时产物是否会被提交（向量库、索引、报告等）；
  4. 打印发布清单与建议命令。

设计原则：**不自动删除你的文件**。--stash 只是移动到 data/_quarantine/ 并在索引里排除，
你可以随时移回来；确认要发布时再把它们从 data/documents 永久移除。
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "data" / "documents"
QUARANTINE_DIR = PROJECT_ROOT / "data" / "_quarantine"
RESUME_DIR = PROJECT_ROOT / "resume"

# 风险特征：(正则, 风险等级, 说明)
#
# 设计教训：早期版本用「维修资料」「内部」这类宽泛词做匹配，结果把自编文档里的
# **免责声明句**（例如"真实诊断请以对应车型的官方维修资料为准"）误判成高风险。
# 现在改为匹配**可验证的来源标识**（URL、平台名、厂商名）与**明确的保密标识**，
# 不再匹配泛化的行业词汇。
RISK_PATTERNS: List[Tuple[str, str, str]] = [
    (
        r"servicenow|saicmotor|dealer\s*portal|授权服务站内部|厂家内部|内部维修资料|禁止外传|仅限内部|机密",
        "高",
        "出现厂商内部/售后服务平台标识或保密标记，公开传播可能同时涉及著作权与商业秘密",
    ),
    (
        r"dongfeng|honda|toyota|nissan|volkswagen|gm\.com|大众|丰田|本田|日产|上汽|一汽|广汽|比亚迪|吉利|长城|蔚来|小鹏|理想",
        "中",
        "涉及具体厂商名，厂商手册/资料的版权通常由厂商保留，公开提供检索建议先获授权",
    ),
    (
        r"999gps|mechanic\s*base|obdcode|code\s*list\s*\.\s*(com|net|org)|转载自|摘自.{0,12}(码表|手册|资料)",
        "中",
        "来源指向第三方汇编站点，对方未必有权许可你转载",
    ),
    (
        r"\.pdf\b|\.docx?\b|扫描件|水印",
        "低",
        "文档本体可能带有版权页或水印，注意不要连带分发原件",
    ),
]

# 这些来源属于低风险，命中厂商名也不算问题（法律文件本身不受著作权保护）
LOW_RISK_HINTS = [
    "三包规定",
    "修理更换退货责任规定",
    "gov.cn",
    "国务院",
    "市场监管总局",
]

SECRET_PATTERNS = [
    (r"sk-[A-Za-z0-9]{16,}", "疑似 API Key（sk- 开头）"),
    (r"AKIA[0-9A-Z]{16}", "疑似 AWS Access Key"),
    (r"ghp_[A-Za-z0-9]{20,}", "疑似 GitHub Token"),
]

# 否定/声明语境指示词。
# 为什么需要：自编的虚构示例文档里会出现"不对应任何真实车企""不包含任何厂商内部资料"
# 这类**声明性**句子，如果只看关键词就会把"声明自己没有风险"的文档误判成高风险。
# 实测踩过：我写的两份虚构码表被自己的检查器报成高风险。
NEGATION_HINTS = [
    "虚构",
    "示例",
    "自编",
    "不对应任何",
    "不代表任何",
    "不包含任何",
    "仅用于",
    "仅用于演示",
    "可自由使用",
    "请勿用于",
    "演示数据",
    "构造",
]

# 判定"否定语境"时看匹配位置前后多大范围
NEGATION_WINDOW = 120

RUNTIME_ARTIFACTS = [
    "data/chroma",
    "data/bm25_index.json",
    "data/registry.json",
    "data/uploads",
    "data/preflight_report.txt",
    "eval/results",
    "__pycache__",
    ".venv",
]


def run_git(args: List[str]) -> Tuple[int, str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode, (result.stdout or "") + (result.stderr or "")
    except FileNotFoundError:
        return 127, "未安装 git 或不在 PATH 中"
    except subprocess.TimeoutExpired:
        return 124, "git 命令超时"


def is_git_repo() -> bool:
    code, _ = run_git(["rev-parse", "--is-inside-work-tree"])
    return code == 0


def tracked_files() -> List[str]:
    code, output = run_git(["ls-files"])
    if code != 0:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def check_secrets() -> List[str]:
    problems: List[str] = []
    print("=" * 72)
    print("1. 密钥与 .env 检查")
    print("=" * 72)

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        print(f"  [OK] 本地 .env 存在（{env_path}），它应当只留在本机")
    else:
        print("  [INFO] 本地没有 .env（可能还没配置）")

    if not is_git_repo():
        print("  [INFO] 当前目录不是 git 仓库，跳过 git 跟踪检查")
        print("         发布前请确认 .env 已在 .gitignore 中（本项目已配置）")
        return problems

    files = tracked_files()
    print(f"  git 已跟踪 {len(files)} 个文件")

    env_tracked = [item for item in files if Path(item).name.startswith(".env") and item != ".env.example"]
    if env_tracked:
        problems.append(
            "以下 .env 类文件已被 git 跟踪，必须先移除并清理历史："
            + "、".join(env_tracked)
            + "（用 git rm --cached 移除跟踪，再清理提交历史）"
        )
        for item in env_tracked:
            print(f"  [严重] 已被跟踪：{item}")
    else:
        print("  [OK] .env 未被 git 跟踪")

    # 在会被提交的文本文件里搜密钥特征
    suspicious: List[str] = []
    for item in files:
        path = PROJECT_ROOT / item
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".py", ".md", ".json", ".yml", ".yaml", ".txt", ".env", ".example", ""}:
            continue
        try:
            if path.stat().st_size > 2 * 1024 * 1024:
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern, label in SECRET_PATTERNS:
            if re.search(pattern, content):
                suspicious.append(f"{item}（{label}）")

    if suspicious:
        problems.append(
            "以下文件中出现疑似密钥特征，请检查并吊销后清理："
            + "、".join(suspicious)
        )
        for item in suspicious:
            print(f"  [严重] {item}")
    else:
        print("  [OK] 已跟踪文件中未发现 sk-/AKIA/ghp_ 形式的密钥")

    return problems


def _has_negation_context(text: str, match: re.Match) -> bool:
    """判断匹配处是否处于"声明/否定"语境（例如"本文件为虚构数据，不对应任何真实车企"）。"""
    start = max(0, match.start() - NEGATION_WINDOW)
    end = min(len(text), match.end() + NEGATION_WINDOW)
    window = text[start:end]
    return any(hint in window for hint in NEGATION_HINTS)


def _scan_risk(content: str) -> List[Tuple[str, str]]:
    """返回命中的 (风险等级, 说明) 列表，已排除声明/否定语境。"""
    head = content[:4000]
    matched: List[Tuple[str, str]] = []
    for pattern, level, reason in RISK_PATTERNS:
        for hit in re.finditer(pattern, head, flags=re.IGNORECASE):
            if _has_negation_context(head, hit):
                continue
            matched.append((level, reason))
            break
    return matched


def check_documents(stash: bool) -> List[str]:
    problems: List[str] = []
    print()
    print("=" * 72)
    print("2. 知识库文档公开传播风险")
    print("=" * 72)

    if not DOCS_DIR.exists():
        print(f"  [INFO] 目录不存在：{DOCS_DIR}")
        return problems

    docs = sorted(
        path for path in DOCS_DIR.rglob("*") if path.is_file() and not path.name.startswith(".")
    )
    if not docs:
        print("  [INFO] 知识库目录为空")
        return problems

    high_risk: List[Path] = []
    for path in docs:
        rel = path.relative_to(DOCS_DIR).as_posix()
        content = ""
        head = ""
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            head = content[:4000]
        except OSError:
            pass

        matched = _scan_risk(content)

        # 法律/政府来源的文档豁免"厂商名"这类中风险
        if any(hint in path.name or hint in head for hint in LOW_RISK_HINTS):
            matched = [(level, reason) for level, reason in matched if level != "中"]

        if not matched:
            print(f"  [低风险] {rel}")
            continue

        worst = "高" if any(level == "高" for level, _ in matched) else "中"
        print(f"  [{worst}风险] {rel}")
        for level, reason in matched:
            print(f"           - ({level}) {reason}")
        if worst == "高":
            high_risk.append(path)

    if high_risk:
        problems.append(
            f"{len(high_risk)} 份文档被判定为高风险："
            + "、".join(path.name for path in high_risk)
        )

    if stash and high_risk:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        print()
        print(f"  --stash 已启用，把高风险文档移动到 {QUARANTINE_DIR}")
        for path in high_risk:
            target = QUARANTINE_DIR / path.name
            shutil.move(str(path), str(target))
            print(f"    已移出：{path.name}")
        print("  提示：移动后请重新入库（python scripts/ingest.py --rebuild），")
        print("        否则向量库里仍保留这些文档的片段。")

    return problems


def check_artifacts() -> List[str]:
    problems: List[str] = []
    print()
    print("=" * 72)
    print("3. 运行时产物检查")
    print("=" * 72)

    if not is_git_repo():
        print("  [INFO] 非 git 仓库，仅列出建议排除的路径：")
        for item in RUNTIME_ARTIFACTS:
            print(f"         {item}")
        return problems

    files = tracked_files()
    leaked = [item for item in RUNTIME_ARTIFACTS if any(f.startswith(item.strip("/")) for f in files)]
    if leaked:
        problems.append("以下运行时产物已被 git 跟踪，建议移除：" + "、".join(leaked))
        for item in leaked:
            print(f"  [警告] 已被跟踪：{item}")
    else:
        print("  [OK] 向量库/索引/评测报告等运行时产物未被跟踪")

    if RESUME_DIR.exists():
        resume_tracked = [item for item in files if item.startswith("resume/")]
        if resume_tracked:
            problems.append(
                "求职材料 resume/ 已被 git 跟踪，里面含面试答案与求职策略，建议不要公开"
            )
            print(f"  [警告] resume/ 下有 {len(resume_tracked)} 个文件被跟踪")

    return problems


def print_checklist(problems: List[str]) -> None:
    print()
    print("=" * 72)
    print("4. 发布清单")
    print("=" * 72)

    steps = [
        "确认 .env 未被提交；如曾提交过，用 git filter-repo 或 BFG 清理历史（仅删除文件不够）",
        "吊销并在控制台重新生成所有 API Key（曾在任何地方明文出现过的都要换）",
        "用可自由使用的文档替换知识库：三包规定全文（法律文件）、自造示例文档；移除厂商手册与内部平台摘录",
        "把 data/documents 里保留的文档重新入库：python scripts/ingest.py --rebuild",
        "确认 data/chroma、data/bm25_index.json、eval/results 等运行时产物未被提交",
        "确认 resume/ 未被提交（本项目 .gitignore 已排除）",
        "README 里写明：示例数据来源与授权情况、项目仅用于技术学习、不代表任何厂商官方口径",
        "可选：加一个 LICENSE（例如 MIT），但注意示例文档的授权情况要单独说明",
    ]
    for index, step in enumerate(steps, start=1):
        print(f"  {index}. {step}")

    print()
    if problems:
        print(f"发现 {len(problems)} 项需要处理：")
        for item in problems:
            print(f"  - {item}")
    else:
        print("未发现阻塞项。仍建议按上面清单逐条人工确认一遍。")


def main() -> int:
    parser = argparse.ArgumentParser(description="作品集发布前检查（脱敏与风险提示）")
    parser.add_argument(
        "--stash",
        action="store_true",
        help="把高风险文档移动到 data/_quarantine/（可逆），默认只检查不改动",
    )
    args = parser.parse_args()

    print("AutoRAG 发布前检查")
    print(f"项目根目录：{PROJECT_ROOT}")
    print()

    problems: List[str] = []
    problems += check_secrets()
    problems += check_documents(args.stash)
    problems += check_artifacts()
    print_checklist(problems)

    print()
    print("提醒：本脚本只做启发式扫描，不能替代你对授权情况的人工判断。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
