#!/usr/bin/env bash
set -Eeuo pipefail

# ===== 可配置 =====
: "${PORT:=8080}"
: "${OPENWEBUI_DATA_DIR:="$HOME/openwebui-data"}"
: "${OPENAI_API_BASE_URL:=https://ollama.com}"
: "${REDIS_URL:=redis://127.0.0.1:6379/0}"
: "${AIOHTTP_CLIENT_TIMEOUT:=1800}"
: "${CORS_ALLOW_ORIGIN:=*}"

# 支持 WEB_CONCURRENCY；显式 WORKERS 优先
: "${WORKERS:=${WEB_CONCURRENCY:-4}}"

DB_PATH="${OPENWEBUI_DATA_DIR}/openwebui.db"
DATABASE_URL_DEFAULT="sqlite:////${DB_PATH}?check_same_thread=false&timeout=30"
: "${DATABASE_URL:=${DATABASE_URL_DEFAULT}}"

banner() {
cat <<'BANNER'
 ██████╗ ██████╗ ███████╗███╗   ██╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║
██║   ██║██████╔╝█████╗  ██╔██╗ ██║
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║
╚██████╔╝██║     ███████╗██║ ╚████║
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝
OpenWebUI launcher
BANNER
}
banner

echo "==> 数据目录: ${OPENWEBUI_DATA_DIR}"
echo "==> 数据库: ${DATABASE_URL}"
echo "==> 模型 API: ${OPENAI_API_BASE_URL}"
echo "==> Redis: ${REDIS_URL}"
echo "==> 端口: ${PORT}"
echo "==> 计划 Workers: ${WORKERS}"

# 目录/文件
mkdir -p "${OPENWEBUI_DATA_DIR}"
touch "${DB_PATH}"

# Redis（Homebrew）
if command -v brew >/dev/null 2>&1; then
  echo "==> 启动 Redis（服务方式）"
  brew services start redis >/dev/null || true
else
  echo "⚠️ 未检测到 Homebrew，请确保 Redis 已运行: ${REDIS_URL}"
fi

# 导出环境变量
export OPENAI_API_BASE_URL DATABASE_URL REDIS_URL AIOHTTP_CLIENT_TIMEOUT CORS_ALLOW_ORIGIN

# ---- 定位前端静态资源目录（给 uvicorn 多进程时使用）----
find_pkg_frontend() {
python3 - <<'PY'
import pkgutil, pathlib, sys
m = pkgutil.get_loader("open_webui")
if not (m and hasattr(m, "get_filename")):
    print("")
    sys.exit(0)
p = pathlib.Path(m.get_filename()).parent / "frontend"
print(str(p) if p.exists() and (p/"index.html").exists() else "")
PY
}

OPENWEBUI_FRONTEND_PATH="$(find_pkg_frontend || true)"
CACHE_FRONTEND="$HOME/.cache/open-webui/frontend"

ensure_frontend() {
  if [ -n "${OPENWEBUI_FRONTEND_PATH}" ] && [ -f "${OPENWEBUI_FRONTEND_PATH}/index.html" ]; then
    return 0
  fi
  if [ -d "${CACHE_FRONTEND}" ] && [ -f "${CACHE_FRONTEND}/index.html" ]; then
    OPENWEBUI_FRONTEND_PATH="${CACHE_FRONTEND}"
    return 0
  fi
  echo "==> 首次运行：预热并拉取前端资源……"
  ( open-webui serve --host 127.0.0.1 --port 0 >/dev/null 2>&1 & PREHEAT_PID=$!; sleep 4; kill "${PREHEAT_PID}" >/dev/null 2>&1 || true )
  OPENWEBUI_FRONTEND_PATH="$(find_pkg_frontend || true)"
  if [ -z "${OPENWEBUI_FRONTEND_PATH}" ] && [ -d "${CACHE_FRONTEND}" ] && [ -f "${CACHE_FRONTEND}/index.html" ]; then
    OPENWEBUI_FRONTEND_PATH="${CACHE_FRONTEND}"
  fi
}

ensure_frontend

if [ -n "${OPENWEBUI_FRONTEND_PATH}" ]; then
  export OPENWEBUI_FRONTEND_PATH
  echo "==> 前端静态文件: ${OPENWEBUI_FRONTEND_PATH}"
else
  echo "⚠️ 未能定位前端目录，若用 uvicorn 多进程可能只暴露 API。"
fi

# ---- 启动：单进程 open-webui，多进程 uvicorn ----
if [ "${WORKERS}" -gt 1 ]; then
  echo "==> 以 uvicorn 多进程方式启动 (${WORKERS} workers)"
  exec python3 -m uvicorn open_webui.main:app \
    --host 0.0.0.0 --port "${PORT}" \
    --workers "${WORKERS}" \
    --proxy-headers --forwarded-allow-ips='*'
else
  echo "==> 以 open-webui serve 单进程方式启动"
  exec open-webui serve --host 0.0.0.0 --port "${PORT}"
fi
