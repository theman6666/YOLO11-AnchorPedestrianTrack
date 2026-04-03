<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Sun, Moon } from 'lucide-vue-next'
import HeroSection from '@/components/layout/HeroSection.vue'
import PanelGrid from '@/components/layout/PanelGrid.vue'
import Card from '@/components/ui/Card.vue'
import StatusMonitor from '@/components/layout/StatusMonitor.vue'

// Reactive state for dark mode
const isDark = ref(false)

// Reactive state for status message (for future Phase 4/5 integration)
const statusMessage = ref('系统就绪。')
const statusIsOk = ref(false)

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
        <!-- Camera Panel -->
        <Card
          title="摄像头实时检测"
          description="实时视频流分析，画面叠加跟踪 ID 与 FPS。"
        >
          <!-- Content added in Phase 3 -->
          <div class="text-gray-400 dark:text-gray-500 text-sm text-center py-8">
            [摄像头面板内容待实现]
          </div>
        </Card>

        <!-- Image Panel -->
        <Card
          title="单张图片检测"
          description="上传单张图片并输出检测标注结果。"
        >
          <!-- Content added in Phase 3 -->
          <div class="text-gray-400 dark:text-gray-500 text-sm text-center py-8">
            [图片面板内容待实现]
          </div>
        </Card>

        <!-- Video Panel -->
        <Card
          title="离线视频分析"
          description="上传视频并输出带跟踪轨迹与标注的结果视频。"
        >
          <!-- Content added in Phase 3 -->
          <div class="text-gray-400 dark:text-gray-500 text-sm text-center py-8">
            [视频面板内容待实现]
          </div>
        </Card>
      </PanelGrid>

      <!-- Status Monitor -->
      <StatusMonitor :message="statusMessage" :isOk="statusIsOk" />
    </div>
  </div>
</template>

<style scoped>
/* All styling via Tailwind utility classes */
</style>
