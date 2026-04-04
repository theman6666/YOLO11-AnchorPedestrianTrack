# Phase 6: Build & Deployment - Context

**Gathered:** 2026-04-04
**Status:** Ready for planning

## Phase Boundary

配置生产优化的构建，使其能够与 Flask 后端一起服务。此阶段专注于构建配置、环境变量系统和 Flask 静态文件服务集成，而不是新功能开发。

**重点：** 部署配置 > 新功能实现。阶段 6 的产出是生产构建设置和部署配置，而不是新的应用功能。

## 实现决策

### 构建输出位置

- **D-01:** 构建输出保持在 `frontend-vue/dist/` 目录 — 不混合到 Flask 代码库
- **D-02:** 开发阶段 Vite 直接服务 dist/ 内容进行预览
- **D-03:** 生产阶段 Flask 配置路由从 dist/ 服务静态文件

### 环境变量配置

- **D-04:** 使用 Vite 内置的 .env 文件模式 — `.env.development` 用于开发，`.env.production` 用于生产
- **D-05:** `VITE_API_BASE_URL` 控制后端 API 地址：
  - 开发：空字符串，使用 Vite 代理（`/video_feed`, `/detect` 等）
  - 生产：完整 URL（如 `http://localhost:5000` 或实际部署地址）
- **D-06:** .env.example 文件作为模板存在，不包含敏感信息

### Flask 集成方式

- **D-07:** Flask 使用 `send_from_directory` 从 `frontend-vue/dist/` 服务静态文件
- **D-08:** API 路由保持 `/video_feed`, `/detect/image`, `/detect/video`, `/results` 前缀
- **D-09:** 所有非 API 请求返回 `index.html`（支持 SPA 路由）

### 构建脚本组织

- **D-10:** 保持现有的 `npm run build` 脚本用于开发和生产构建
- **D-11:** Vite 根据 `NODE_ENV` 或 `--mode` 自动选择正确的 .env 文件
- **D-12:** 不创建单独的 build:dev/build:prod 脚本

### 生产构建优化

- **D-13:** 启用 Vite 的默认优化（代码分割、压缩、tree-shaking）
- **D-14:** 使用时间戳或哈希文件名进行缓存破坏（Vite 默认行为）
- **D-15:** 确保所有资源路径相对正确，不依赖开发服务器的代理

### Claude 自由裁量

- Flask 路由的具体实现细节（蓝图 vs 直接路由）
- 是否添加构建后的验证步骤（如文件存在性检查）
- 生产环境的额外安全头（CORS、CSP 等）

## 规范引用

**下游代理在规划或实现前必须阅读这些内容。**

### 现有构建配置
- `frontend-vue/vite.config.ts` — 当前的 Vite 配置（开发服务器代理设置）
- `frontend-vue/package.json` — 构建脚本和依赖
- `frontend-vue/.env.example` — 环境变量模板
- `frontend-vue/tsconfig.*.json` — TypeScript 配置文件

### Flask 后端
- `src/run/app.py` — Flask 应用入口，需要添加静态文件服务路由

### 前端 API 客户端
- `frontend-vue/src/api/client.ts` — Axios 客户端，使用 `import.meta.env.VITE_API_BASE_URL`

### 需求映射
- `.planning/REQUIREMENTS.md` — BUILD-01 至 BUILD-04 定义成功标准

### 阶段 5 验证输出
- `.planning/phases/05-feature-implementation/test-cases/00-test-environment-setup.md` — 测试环境设置说明

## 现有代码见解

### 可复用资产

- **Vite 配置**: 已配置开发服务器代理到 Flask 后端
- **构建脚本**: `npm run build` 和 `npm run build-only` 已存在
- **环境变量模板**: `.env.example` 已定义 `VITE_API_BASE_URL`
- **Axios 客户端**: 已使用 `import.meta.env.VITE_API_BASE_URL` 获取配置

### 已建立的模式

- **开发模式**: Vite 开发服务器在 5173 端口，代理 API 请求到 5000 端口
- **TypeScript 配置**: 使用项目引用和严格的类型检查
- **Tailwind CSS**: PostCSS 配置已就绪，支持生产构建

### 集成点

- **Flask 应用**: 需要添加路由从 `frontend-vue/dist/` 服务 `index.html` 和静态资源
- **API 客户端**: 需要在生产环境直接连接到 Flask API（不经过代理）
- **构建输出**: dist/ 目录包含 index.html 和 assets/ 文件夹

## 具体想法

- 开发环境 .env 文件保持 `VITE_API_BASE_URL=` 空字符串，触发 Vite 代理
- 生产环境 .env 文件设置 `VITE_API_BASE_URL=http://localhost:5000` 或实际部署地址
- Flask 添加 catch-all 路由，对于非 `/api`, `/video_feed`, `/detect`, `/results` 开头的请求返回 index.html
- 考虑添加构建后的简单验证脚本，检查 dist/index.html 是否存在

## 延迟的想法

- **Nginx/Apache 集成** — 不在此阶段实现，Flask 直接服务静态文件已足够
- **Docker 容器化** — 延迟到后续阶段，不属于阶段 6 范围
- **CI/CD 流水线** — 不在阶段 6 实现，仅配置本地构建和部署
- **多环境配置** — 仅支持开发和生产，不添加 staging 环境
- **HTTPS/SSL 配置** — 不在阶段 6 处理

---

*阶段: 06-build-deployment*
*上下文收集时间: 2026-04-04*
