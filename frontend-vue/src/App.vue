<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Sun, Moon } from 'lucide-vue-next'
import HeroSection from '@/components/layout/HeroSection.vue'
import PanelGrid from '@/components/layout/PanelGrid.vue'
import StatusMonitor from '@/components/layout/StatusMonitor.vue'
import CameraPanel from '@/components/panels/CameraPanel.vue'
import ImagePanel from '@/components/panels/ImagePanel.vue'
import VideoPanel from '@/components/panels/VideoPanel.vue'

// Reactive state for dark mode
const isDark = ref(false)

// Reactive state for status message
const statusMessage = ref('系统就绪。')
const statusIsOk = ref(false)

// Reactive state for processing flags (per D-13)
const processing = ref({
  camera: false,
  image: false,
  video: false,
})

// Reactive state for panel results (for Phase 4 integration)
const cameraStreamUrl = ref<string | undefined>(undefined)
const imageResultUrl = ref<string | undefined>(undefined)
const imagePersonCount = ref<number | undefined>(undefined)
const imageErrorMessage = ref<string | undefined>(undefined)
const videoResultUrl = ref<string | undefined>(undefined)
const videoStats = ref<{ frames: number; avg_fps: number } | undefined>(undefined)
const videoErrorMessage = ref<string | undefined>(undefined)

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

// Camera panel event handlers (Phase 4 will implement API calls)
const handleCameraStart = (cameraId: number) => {
  console.log('Start camera:', cameraId)
  statusMessage.value = `摄像头 ${cameraId} 已启动。`
  statusIsOk.value = true
  // Phase 4: Set cameraStreamUrl.value = `/video_feed?camera_id=${cameraId}`
}

const handleCameraStop = () => {
  console.log('Stop camera')
  statusMessage.value = '摄像头已停止。'
  statusIsOk.value = false
  cameraStreamUrl.value = undefined
  // Phase 4: Clear stream by setting src to empty string
}

// Image panel event handler (Phase 4 will implement API call)
const handleImageDetect = async (file: File) => {
  console.log('Detect image:', file.name)
  statusMessage.value = '正在进行图片检测...'
  statusIsOk.value = false
  processing.value.image = true
  imageErrorMessage.value = undefined

  try {
    // Phase 4: Implement API call to /detect/image
    // const formData = new FormData()
    // formData.append('file', file)
    // const response = await fetch('/detect/image', { method: 'POST', body: formData })
    // const data = await response.json()

    // Simulate API call for now
    await new Promise(resolve => setTimeout(resolve, 1000))

    // Phase 4: Update with real data
    // imageResultUrl.value = data.image_url
    // imagePersonCount.value = data.count
    // statusMessage.value = data.message || '图片检测完成。'
    // statusIsOk.value = true

    statusMessage.value = '图片检测功能将在 Phase 4 实现。'
    statusIsOk.value = true
  } catch (error) {
    statusMessage.value = '图片检测出错。'
    statusIsOk.value = false
    imageErrorMessage.value = '图片检测出错。'
  } finally {
    processing.value.image = false
  }
}

// Video panel event handler (Phase 4 will implement API call)
const handleVideoDetect = async (file: File) => {
  console.log('Detect video:', file.name)
  statusMessage.value = '正在进行视频检测，耗时可能较长，请稍候。'
  statusIsOk.value = false
  processing.value.video = true
  videoErrorMessage.value = undefined

  try {
    // Phase 4: Implement API call to /detect/video
    // const formData = new FormData()
    // formData.append('file', file)
    // const response = await fetch('/detect/video', { method: 'POST', body: formData })
    // const data = await response.json()

    // Simulate API call for now
    await new Promise(resolve => setTimeout(resolve, 2000))

    // Phase 4: Update with real data
    // videoResultUrl.value = data.video_url
    // videoStats.value = data.stats
    // statusMessage.value = data.message || '视频检测完成。'
    // statusIsOk.value = true

    statusMessage.value = '视频检测功能将在 Phase 4 实现。'
    statusIsOk.value = true
  } catch (error) {
    statusMessage.value = '视频检测出错。'
    statusIsOk.value = false
    videoErrorMessage.value = '视频检测出错。'
  } finally {
    processing.value.video = false
  }
}
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
        <CameraPanel
          :processing="processing.camera"
          :stream-url="cameraStreamUrl"
          @start="handleCameraStart"
          @stop="handleCameraStop"
        />

        <!-- Image Panel -->
        <ImagePanel
          :processing="processing.image"
          :result-url="imageResultUrl"
          :person-count="imagePersonCount"
          :error-message="imageErrorMessage"
          @detect="handleImageDetect"
        />

        <!-- Video Panel -->
        <VideoPanel
          :processing="processing.video"
          :result-url="videoResultUrl"
          :video-stats="videoStats"
          :error-message="videoErrorMessage"
          @detect="handleVideoDetect"
        />
      </PanelGrid>

      <!-- Status Monitor -->
      <StatusMonitor :message="statusMessage" :isOk="statusIsOk" />
    </div>
  </div>
</template>

<style scoped>
/* All styling via Tailwind utility classes */
</style>
