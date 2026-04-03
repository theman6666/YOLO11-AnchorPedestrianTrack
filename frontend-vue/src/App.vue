<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Sun, Moon } from 'lucide-vue-next'
import HeroSection from '@/components/layout/HeroSection.vue'
import PanelGrid from '@/components/layout/PanelGrid.vue'

// Reactive state for dark mode
const isDark = ref(false)

// Toggle dark mode
const toggleDarkMode = () => {
  document.documentElement.classList.toggle('dark')
  isDark.value = document.documentElement.classList.contains('dark')
}

// Initialize dark mode on mount (default to dark per D-01)
onMounted(() => {
  document.documentElement.classList.add('dark')
  isDark.value = true
})
</script>

<template>
  <div
    class="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors"
    style="background-image: linear-gradient(90deg, rgba(31,78,121,0.03) 1px, transparent 1px), linear-gradient(rgba(31,78,121,0.03) 1px, transparent 1px); background-size: 36px 36px;"
  >
    <div class="max-w-[1180px] mx-auto p-[28px_20px_36px]">
      <!-- Dark mode toggle button (fixed top-right) -->
      <button
        @click="toggleDarkMode"
        class="fixed top-4 right-4 z-50 p-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-md hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
        aria-label="Toggle dark mode"
      >
        <Sun v-if="isDark" :size="20" class="text-yellow-500" />
        <Moon v-else :size="20" class="text-gray-600" />
      </button>

      <!-- Hero Section -->
      <HeroSection />

      <!-- 3-Panel Grid -->
      <PanelGrid>
        <!-- Camera Panel (placeholder) -->
        <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl shadow-lg p-4">
          <h2 class="text-xl font-semibold text-gray-900 dark:text-white">摄像头实时检测</h2>
          <div class="text-sm text-gray-600 dark:text-gray-400 mt-1.5 mb-3.5">实时视频流分析，画面叠加跟踪 ID 与 FPS。</div>
          <!-- Content added in Phase 3 -->
        </div>

        <!-- Image Panel (placeholder) -->
        <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl shadow-lg p-4">
          <h2 class="text-xl font-semibold text-gray-900 dark:text-white">单张图片检测</h2>
          <div class="text-sm text-gray-600 dark:text-gray-400 mt-1.5 mb-3.5">上传单张图片并输出检测标注结果。</div>
          <!-- Content added in Phase 3 -->
        </div>

        <!-- Video Panel (placeholder) -->
        <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl shadow-lg p-4">
          <h2 class="text-xl font-semibold text-gray-900 dark:text-white">离线视频分析</h2>
          <div class="text-sm text-gray-600 dark:text-gray-400 mt-1.5 mb-3.5">上传视频并输出带跟踪轨迹与标注的结果视频。</div>
          <!-- Content added in Phase 3 -->
        </div>
      </PanelGrid>
    </div>
  </div>
</template>

<style scoped>
/* All styling via Tailwind utility classes */
</style>
