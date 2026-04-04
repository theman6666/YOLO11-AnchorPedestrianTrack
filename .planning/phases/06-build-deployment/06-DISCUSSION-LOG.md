# Phase 6: Build & Deployment - Discussion Log

> **审计跟踪仅供参考。** 不要作为规划、研究或执行的输入。
> 决策已记录在 CONTEXT.md 中 — 此日志保留考虑的替代方案。

**Date:** 2026-04-04
**Phase:** 06-build-deployment
**Areas discussed:** 构建位置, 环境配置, Flask 集成, 构建脚本

---

## 构建位置

| Option | Description | Selected |
|--------|-------------|----------|
| Flask static 文件夹 | 构建输出到 src/static 或 frontend/static，Flask 直接服务。优点：Flask 原生支持；缺点：混合前后端代码 | |
| 独立的 dist/ 文件夹 | 保持构建输出在 frontend-vue/dist/，Flask 配置路由代理。优点：前后端代码分离；缺点：需要额外路由配置 | ✓ |
| 构建后复制 | 先构建到 dist/，然后复制到 Flask static。优点：两者兼顾；缺点：需要额外的复制步骤 | |

**User's choice:** 独立的 dist/ 文件夹
**Notes:** 保持前后端代码分离，使用 Flask 路由从 dist/ 服务文件

---

## 环境配置

| Option | Description | Selected |
|--------|-------------|----------|
| .env 文件（推荐） | 开发用 .env.development，生产用 .env.production。Vite 自动选择。优点：Vite 内置支持；缺点：需要多个文件 | ✓ |
| 构建时替换 | 使用 vite define 插件在构建时注入配置。优点：单文件；缺点：重新构建才能更改 | |
| 运行时配置 | 加载 config.json 文件。优点：无需重新构建；缺点：需要额外的加载逻辑 | |

**User's choice:** .env 文件（推荐）
**Notes:** 使用 Vite 内置的 .env 文件模式，开发时使用代理，生产时直接连接 API

---

## Flask 集成

| Option | Description | Selected |
|--------|-------------|----------|
| Flask static 文件夹 | 使用 send_from_directory 或 static_folder。优点：Flask 原生方式；缺点：需要构建输出到 static 目录 | ✓ |
| 专用路由 | 创建 /api 前缀的 API 路由，其他所有请求返回 index.html。优点：支持 SPA 路由；缺点：需要自定义路由 | |
| 反向代理 | Flask 只提供 API，前端由 Nginx/Apache 服务。优点：生产最佳实践；缺点：需要额外服务器配置 | |

**User's choice:** Flask static 文件夹
**Notes:** 使用 send_from_directory 从 dist/ 服务文件，所有非 API 请求返回 index.html

---

## 构建脚本

| Option | Description | Selected |
|--------|-------------|----------|
| 保持现有 | 使用 npm run build 进行所有构建。优点：简单；缺点：开发/生产使用相同配置 | ✓ |
| 分离脚本 | 添加 build:dev 和 build:prod 脚本。优点：明确区分环境；缺点：更多脚本需要维护 | |
| 环境模式 | 使用 VITE_ENV 环境变量控制构建行为。优点：灵活；缺点：需要更多配置逻辑 | |

**User's choice:** 保持现有
**Notes:** Vite 根据 NODE_ENV 自动选择正确的 .env 文件

---

## Claude's Discretion

无 — 用户对所有决策都做出了明确选择

## Deferred Ideas

- Nginx/Apache 集成 — 使用反向代理，属于高级部署配置
- Docker 容器化 — 需要单独的阶段处理
- CI/CD 流水线 — 超出阶段 6 范围
- 多环境配置（staging） — 仅支持开发和生产
- HTTPS/SSL 配置 — 安全相关，后续阶段处理
