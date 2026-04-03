<script setup lang="ts">
import { ref } from 'vue'
import Card from '@/components/ui/Card.vue'
import PreviewArea from '@/components/ui/PreviewArea.vue'

interface Props {
  title?: string
  description?: string
  processing?: boolean
  streamUrl?: string
  cameraId?: number
}

withDefaults(defineProps<Props>(), {
  title: '摄像头实时检测',
  description: '实时视频流分析，画面叠加跟踪 ID 与 FPS。',
  processing: false,
  streamUrl: undefined,
  cameraId: undefined,
})

const emit = defineEmits<{
  start: [cameraId: number]
  stop: []
}>()

const localCameraId = ref<number>(0)

const handleStart = () => {
  const id = localCameraId.value || 0
  emit('start', id)
}

const handleStop = () => {
  emit('stop')
}
</script>

<template>
  <Card :title="title" :description="description">
    <div class="space-y-3">
      <!-- Camera ID Input -->
      <div>
        <label
          for="cameraId"
          class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5"
        >
          摄像头编号
        </label>
        <input
          id="cameraId"
          v-model.number="localCameraId"
          type="number"
          min="0"
          :disabled="processing"
          class="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-accent-500 focus:border-accent-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          placeholder="0"
        />
      </div>

      <!-- Action Buttons -->
      <div class="flex gap-2">
        <!-- Start Button (Primary) -->
        <button
          @click="handleStart"
          :disabled="processing"
          class="flex-1 bg-accent-600 hover:bg-accent-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium py-2 px-4 rounded-lg transition-colors"
        >
          {{ processing ? '处理中...' : '启动摄像头' }}
        </button>

        <!-- Stop Button (Secondary) -->
        <button
          @click="handleStop"
          :disabled="processing"
          class="flex-1 bg-white dark:bg-gray-800 text-accent-600 dark:text-accent-500 border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium py-2 px-4 rounded-lg transition-colors"
        >
          停止摄像头
        </button>
      </div>

      <!-- Preview Area -->
      <PreviewArea>
        <img
          v-if="streamUrl"
          :src="streamUrl"
          alt="摄像头画面"
          class="w-full h-full object-contain"
        />
        <span v-else class="text-gray-400 dark:text-gray-500">
          预览区域
        </span>
      </PreviewArea>
    </div>
  </Card>
</template>

<style scoped>
/* All styling via Tailwind utility classes */
</style>
