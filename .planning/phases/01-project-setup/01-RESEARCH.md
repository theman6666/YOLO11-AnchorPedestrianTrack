# Phase 1: Project Setup - Research

**Researched:** 2026-04-03
**Domain:** Vue 3 + Vite frontend project initialization
**Confidence:** HIGH

## Summary

Phase 1 focuses on establishing a modern Vue 3 + Vite development environment with TypeScript support, Tailwind CSS for styling, and proper configuration for API communication with the existing Flask backend. The research confirms that the Vue 3 ecosystem is mature and stable, with clear best practices for project initialization.

**Primary recommendation:** Use `npm create vue@latest` for scaffold generation, then manually add Tailwind CSS and configure Vite proxy for Flask backend integration. This approach provides the most flexibility while following Vue 3 official recommendations.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SETUP-01 | Vue 3 + Vite project initialized with TypeScript support | Vue 3.5.32 + Vite 6.0.3 with @vitejs/plugin-vue 6.0.5 provides stable TypeScript support. Official scaffold command available. |
| SETUP-02 | Tailwind CSS configured with dark mode support | Tailwind CSS 4.2.2 with dark mode configuration. Two approaches available: traditional `darkMode: 'class'` or new v4 Vite plugin with custom variants. |
| SETUP-03 | Lucide-Vue-Next icons library integrated | lucide-vue-next 1.0.0 provides Vue 3 components. Install via npm and import individual icons as components. |
| SETUP-04 | Axios installed and configured for API communication | Axios 1.14.0 stable. Best practices: create dedicated instance with base URL, interceptors for error handling. |
| SETUP-05 | Development server configured to proxy Flask backend API | Vite server.proxy configuration to forward /video_feed, /detect/*, /results/* to Flask on port 5000. |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Vue | 3.5.32 | Progressive JavaScript framework | Latest stable, excellent TypeScript support, Composition API |
| Vite | 6.0.3 | Build tool and dev server | Lightning-fast HMR, native ESM, official Vue recommendation |
| TypeScript | 6.0.2 | Static typing | Enhanced developer experience, better IDE support |
| @vitejs/plugin-vue | 6.0.5 | Vue 3 integration for Vite | Official plugin, seamless SFC support |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Tailwind CSS | 4.2.2 | Utility-first CSS framework | Rapid styling, dark mode support, industrial aesthetic |
| Axios | 1.14.0 | HTTP client | API communication, interceptors, error handling |
| lucide-vue-next | 1.0.0 | Icon library | Consistent icon set, tree-shakeable |
| Autoprefixer | 10.4.27 | CSS vendor prefixes | Tailwind dependency, browser compatibility |
| PostCSS | 8.5.8 | CSS transformation | Tailwind dependency, modern CSS processing |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Vue CLI | Vite | Vite is faster, modern, officially recommended by Vue team |
| Axios | Fetch API | Axios provides interceptors, request cancellation, better error handling |
| Tailwind | CSS Modules | Tailwind faster for development, consistent dark mode, utility-first |

**Installation:**
```bash
# Create Vue 3 + TypeScript project
npm create vue@latest frontend-vue

# Navigate and install core dependencies
cd frontend-vue
npm install

# Install additional dependencies
npm install axios lucide-vue-next
npm install -D tailwindcss autoprefixer postcss
```

**Version verification:** All package versions verified via npm view command on 2026-04-03.

## Architecture Patterns

### Recommended Project Structure
```
frontend-vue/
├── public/              # Static assets
├── src/
│   ├── assets/          # Images, styles, fonts
│   ├── components/      # Vue components (organized by feature)
│   ├── composables/     # Reusable composition functions
│   ├── api/             # API client configuration and endpoints
│   ├── types/           # TypeScript type definitions
│   ├── App.vue          # Root component
│   └── main.ts          # Application entry point
├── index.html           # HTML template
├── vite.config.ts       # Vite configuration (proxy, plugins)
├── tailwind.config.js   # Tailwind CSS configuration
├── tsconfig.json        # TypeScript configuration
├── package.json         # Dependencies and scripts
└── README.md            # Project documentation
```

### Pattern 1: Vue 3 Composition API with `<script setup lang="ts">`
**What:** Modern Vue 3 syntax combining Composition API with TypeScript and compile-time optimizations
**When to use:** All new components - this is the recommended pattern for Vue 3
**Example:**
```typescript
// Source: Vue 3 official documentation
<script setup lang="ts">
import { ref, computed } from 'vue'

interface Props {
  title: string
  count?: number
}

const props = withDefaults(defineProps<Props>(), {
  count: 0
})

const isActive = ref(false)
const displayText = computed(() => `${props.title}: ${props.count}`)
</script>
```

### Pattern 2: Axios Instance Configuration
**What:** Centralized Axios instance with base URL and interceptors
**When to use:** All API communication - provides consistent error handling and configuration
**Example:**
```typescript
// Source: Axios best practices (Mintlify guide)
// src/api/client.ts
import axios from 'axios'

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth tokens or logging
    return config
  },
  (error) => Promise.reject(error)
)

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Centralized error handling
    console.error('API Error:', error)
    return Promise.reject(error)
  }
)
```

### Pattern 3: Vite Proxy Configuration
**What:** Development server proxy to forward API requests to Flask backend
**When to use:** Development only - production builds will be served by Flask
**Example:**
```typescript
// Source: Vite documentation
// vite.config.ts
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      // Proxy video feed endpoint
      '/video_feed': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      // Proxy detection endpoints
      '/detect': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      // Proxy result files
      '/results': {
        target: 'http://localhost:5000',
        changeOrigin: true
      }
    }
  }
})
```

### Pattern 4: Tailwind Dark Mode Configuration
**What:** Configure Tailwind to support manual dark mode toggling via class strategy
**When to use:** Need user-controlled dark mode switching (industrial aesthetic requirement)
**Example:**
```javascript
// Source: Tailwind CSS dark mode documentation
// tailwind.config.js
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  darkMode: 'class', // Manual control via .dark class
  theme: {
    extend: {
      colors: {
        // Industrial dark mode palette
        gray: {
          850: '#1f2937',
          900: '#111827',
          950: '#030712',
        }
      }
    }
  },
  plugins: []
}
```

### Anti-Patterns to Avoid
- **Mixing Options API and Composition API**: Choose one approach per component. For new code, always use Composition API with `<script setup>`.
- **Hardcoding API URLs**: Use environment variables and Vite proxy for development flexibility.
- **Global CSS imports in components**: Import global styles once in main.ts, use scoped styles in components.
- **Direct DOM manipulation**: Use Vue's reactive system and template refs instead of `document.querySelector`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HTTP requests | Custom fetch wrappers | Axios with interceptors | Error handling, request cancellation, timeout management, upload progress |
| State management | Custom event bus | Pinia (future phases) | DevTools integration, TypeScript support, modular stores |
| Icons | Custom SVG components | lucide-vue-next | Consistent design, tree-shakeable, 1000+ icons |
| Hot Module Replacement | Custom file watchers | Vite HMR | Instant updates, ecosystem standard, framework-aware |
| CSS preprocessing | Custom build scripts | Tailwind + PostCSS | Industry standard, autoprefixer, purge optimization |
| TypeScript compilation | Custom tsc scripts | Vite plugin | Faster, on-demand compilation, better HMR |

**Key insight:** The Vue 3 ecosystem has mature solutions for all common problems. Custom implementations increase maintenance burden and miss out on ecosystem improvements and tooling support.

## Common Pitfalls

### Pitfall 1: Flask Backend CORS Issues
**What goes wrong:** Browser blocks requests from Vite dev server (typically port 5173) to Flask backend (port 5000) due to CORS policy
**Why it happens:** Flask doesn't allow cross-origin requests by default, and Vite dev server runs on different port
**How to avoid:** Use Vite's proxy configuration (Pattern 3) to make requests appear same-origin. Never disable CORS in production.
**Warning signs:** Browser console shows "Access-Control-Allow-Origin" errors, network tab shows preflight OPTIONS requests failing

### Pitfall 2: Video Stream Not Updating
**What goes wrong:** Camera feed displays initial frame but never updates
**Why it happens:** Multipart stream handling requires special Axios configuration or direct `<img>` tag usage with cache-busting
**How to avoid:** Use direct `<img>` tag with `/video_feed?camera_id=X&t=${timestamp}` URL pattern. Don't use Axios for streaming video.
**Warning signs:** Static image instead of video, no errors in console

### Pitfall 3: Tailwind Dark Mode Not Working
**What goes wrong:** Dark mode classes don't apply even when configured
**Why it happens:** Missing `darkMode: 'class'` configuration or not toggling `.dark` class on `<html>` element
**How to avoid:** Verify tailwind.config.js has `darkMode: 'class'`, implement dark mode toggle component that adds/removes `.dark` class on document.documentElement
**Warning signs:** Dark mode classes (dark:bg-gray-900) have no effect, light theme always shows

### Pitfall 4: TypeScript Path Resolution Issues
**What goes wrong:** Import errors for @/ aliases or absolute imports
**Why it happens:** Vite and TypeScript use different configuration for path mapping
**How to avoid:** Configure both vite.config.ts (resolve.alias) and tsconfig.json (compilerOptions.paths) with matching paths
**Warning signs:** Red squiggles in IDE despite code working, "Cannot find module" errors

### Pitfall 5: Development vs Production Environment Confusion
**What goes wrong:** Code works in dev but breaks in production build
**Why it happens:** Vite proxy only works in development. Production builds need Flask to serve static files
**How to avoid:** Use environment variables (VITE_API_BASE_URL) to distinguish dev/prod. Configure Flask to serve built Vue app as static files in production.
**Warning signs:** API requests return 404 in production, different behavior between dev and build

## Code Examples

Verified patterns from official sources:

### Vue 3 Component with TypeScript
```typescript
// Source: Vue 3 Composition API documentation
// src/components/StatusMonitor.vue
<script setup lang="ts">
import { ref, watch } from 'vue'

interface StatusMessage {
  text: string
  type: 'info' | 'success' | 'error'
}

const props = defineProps<{
  status?: StatusMessage
}>()

const emit = defineEmits<{
  (e: 'clear'): void
}>()

const localStatus = ref<StatusMessage | undefined>(props.status)

watch(() => props.status, (newStatus) => {
  localStatus.value = newStatus
})
</script>

<template>
  <div v-if="localStatus" :class="['status', `status--${localStatus.type}`]">
    {{ localStatus.text }}
    <button @click="emit('clear')" class="status__close">×</button>
  </div>
</template>

<style scoped>
.status {
  /* component styles */
}
</style>
```

### Vite Configuration with TypeScript
```typescript
// Source: Vite TypeScript documentation
// vite.config.ts
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  },
  server: {
    port: 5173,
    proxy: {
      '/video_feed': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        ws: true // Enable WebSocket proxy if needed
      },
      '/detect': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/results': {
        target: 'http://localhost:5000',
        changeOrigin: true
      }
    }
  }
})
```

### Axios Instance with TypeScript Types
```typescript
// Source: Axios TypeScript usage
// src/api/client.ts
import axios, { AxiosError, AxiosResponse } from 'axios'

// API response types from Flask backend
interface DetectionResponse {
  ok: boolean
  count?: number
  image_url?: string
  video_url?: string
  stats?: {
    frames: number
    avg_fps: number
  }
  message: string
}

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  timeout: 30000,
  headers: {
    'Content-Type': 'multipart/form-data' // For file uploads
  }
})

// Type-safe request helper
export async function detectImage(file: File): Promise<DetectionResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const response: AxiosResponse<DetectionResponse> = await apiClient.post(
    '/detect/image',
    formData
  )

  return response.data
}
```

### Tailwind CSS Integration
```css
/* Source: Tailwind CSS Vite plugin documentation */
/* src/assets/main.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

/* Custom dark mode styles */
@layer components {
  .btn-primary {
    @apply px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors;
  }

  .dark .btn-primary {
    @apply bg-blue-500 hover:bg-blue-600;
  }
}

/* Industrial dark mode palette */
@layer base {
  :root {
    --color-bg-primary: #f8fafc;
    --color-bg-secondary: #ffffff;
    --color-text-primary: #0f172a;
  }

  .dark {
    --color-bg-primary: #0f172a;
    --color-bg-secondary: #1e293b;
    --color-text-primary: #f1f5f9;
  }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Vue CLI | Vite | 2022-2023 | Vite is now official Vue recommendation, 10-100x faster HMR |
| Options API | Composition API + `<script setup>` | Vue 3.2+ (2021) | Less boilerplate, better TypeScript inference, tree-shaking |
| Webpack | Vite (Rollup-based) | 2021+ | Faster dev server, optimized production builds, native ESM |
| Vuex | Pinia | Vue 3 official (2022) | Better TypeScript support, simpler API, devtools integration |
| CSS Modules/Sass | Tailwind CSS | 2019+ | Rapid development, consistent design system, smaller bundles |

**Deprecated/outdated:**
- Vue 2: End of life December 2023, no security updates
- Vue CLI: In maintenance mode, Vite is recommended for new projects
- class-based components: Experimental in Vue 2, not part of Vue 3
- Vuex: Replaced by Pinia for Vue 3 state management
- `filter` option: Removed in Vue 3, use methods or computed properties instead

## Open Questions

1. **Flask CORS Configuration**
   - What we know: Vite proxy handles development CORS. Production will serve Vue build from Flask static files.
   - What's unclear: Whether Flask CORS middleware needs configuration for edge cases (e.g., direct API access).
   - Recommendation: Test API calls through Vite proxy first. Add Flask-CORS extension only if direct browser-to-Flask access is needed in production.

2. **Build Output Integration**
   - What we know: Vite outputs to `dist/` by default. Flask needs to serve these as static files.
   - What's unclear: Exact Flask route configuration for SPA routing (handling client-side routes).
   - Recommendation: Configure Flask to serve index.html for unmatched routes, implement catch-all route in Phase 6 (Build & Deployment).

3. **Video Stream Performance**
   - What we know: Existing HTML uses direct `<img>` tag with multipart stream. Vite proxy should handle this.
   - What's unclear: Whether Vite proxy adds latency or buffering to video stream compared to direct Flask access.
   - Recommendation: Benchmark video stream performance in Phase 1. If proxy adds significant latency, consider direct camera access to Flask port for video only.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Node.js | Vue 3 + Vite | ✓ | (Assume installed) | — |
| npm | Package management | ✓ | (Assume installed) | — |
| Python | Flask backend (existing) | ✓ | (Assume installed) | — |
| Flask | Backend API (existing) | ✓ | (Already running) | — |

**Assumptions:**
- Node.js and npm are available (required for Vue 3 development)
- Python and Flask are already installed and working (existing backend)
- No additional tools needed for Phase 1

**Verification needed:**
- Node.js version >= 18.0.0 (Vue 3 + Vite minimum requirement)
- npm version >= 9.0.0
- Flask backend is accessible on port 5000

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Vitest (planned for Phase 6, not required for Phase 1) |
| Config file | None - Phase 1 focuses on project setup |
| Quick run command | `npm run dev` - verify dev server starts |
| Full suite command | `npm run build` - verify build succeeds |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SETUP-01 | Vue 3 + Vite project initializes and runs | smoke | `npm run dev` | ✅ Wave 0 |
| SETUP-02 | Tailwind CSS styles load correctly | visual | Manual browser check | ✅ Wave 0 |
| SETUP-03 | Lucide icons render without errors | smoke | Component import test | ✅ Wave 0 |
| SETUP-04 | Axios can make API requests | integration | `npm run dev` + API call | ✅ Wave 0 |
| SETUP-05 | Vite proxy forwards to Flask | integration | Browser DevTools Network tab | ✅ Wave 0 |

### Sampling Rate
- **Per task commit:** `npm run dev` - verify dev server still starts
- **Per wave merge:** `npm run build` - verify production build succeeds
- **Phase gate:** All 5 requirements verified manually (dev server runs, styles work, icons render, API connects, proxy works)

### Wave 0 Gaps
- [ ] `npm run dev` command - verifies Vite dev server starts without errors
- [ ] `npm run build` command - verifies production build succeeds
- [ ] Manual browser test - verify Tailwind styles apply and dark mode class toggles
- [ ] Manual API test - verify Axios can communicate with proxied Flask endpoints
- [ ] Component smoke test - simple component rendering a Lucide icon to verify integration

## Sources

### Primary (HIGH confidence)
- Vue 3 official documentation - https://vuejs.org/guide/introduction.html (Composition API, `<script setup>`, TypeScript support)
- Vite official documentation - https://vitejs.dev/guide/ (proxy configuration, plugins, TypeScript)
- Tailwind CSS official documentation - https://tailwindcss.com/docs/installation (dark mode configuration, Vite integration)
- Axios documentation - https://axios-http.com/docs/intro (instance configuration, interceptors)
- npm package registry - Verified all package versions via `npm view` command on 2026-04-03

### Secondary (MEDIUM confidence)
- [Lucide for Vue - Official Guide](https://lucide.dev/guide/vue) - Icon library integration
- [HowTo: Toggle dark mode with TailwindCSS + Vue3 + Vite](https://stackoverflow.com/questions/71871232/howto-toggle-dark-mode-with-tailwindcss-vue3-vite) - Dark mode implementation patterns
- [Dark mode - Core concepts - Tailwind CSS](https://tailwindcss.com/docs/dark-mode) - Official dark mode documentation
- [How To Create a React + Flask Project](https://blog.miguelgrinberg.com/post/how-to-create-a-react--flask-project) - Flask + frontend integration patterns
- [Axios + Vue.js 3 + Pinia Configuration Guide](https://medium.com/@bugintheconsole/axios-vue-js-3-pinia-a-comfy-configuration-you-can-consider-for-an-api-rest-a6005c356dcd) - Axios best practices
- [Axios Best Practices Guide](https://www.mintlify.com/axios/axios/guides/best-practices) - Dedicated instances, interceptors
- [Tell Flask it is Behind a Proxy — Flask Documentation](https://flask.palletsprojects.com/en/stable/deploying/proxy_fix/) - Flask proxy configuration

### Tertiary (LOW confidence)
- Various Stack Overflow threads and blog posts - Used for cross-verification only
- General Vue 3 tutorials and guides - Cross-referenced with official docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All versions verified via npm, stable releases, official documentation available
- Architecture: HIGH - Vue 3 and Vite patterns well-documented, official best practices clear
- Pitfalls: MEDIUM - Based on common Vue 3 + Flask integration issues, but some may not apply to this specific project

**Research date:** 2026-04-03
**Valid until:** 2026-05-03 (30 days - stable ecosystem, but versions may update)
