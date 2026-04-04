<script setup lang="ts">
import { ref } from 'vue'
import Card from '@/components/ui/Card.vue'
import PreviewArea from '@/components/ui/PreviewArea.vue'
import FileInput from '@/components/ui/FileInput.vue'

interface Props {
  title?: string
  description?: string
  processing?: boolean
  resultUrl?: string
  personCount?: number
  errorMessage?: string
}

withDefaults(defineProps<Props>(), {
  title: '单张图片检测',
  description: '上传单张图片并输出检测标注结果。',
  processing: false,
  resultUrl: undefined,
  personCount: undefined,
  errorMessage: undefined,
})

const emit = defineEmits<{
  detect: [file: File]
}>()

const selectedFile = ref<File | null>(null)
const fileInputRef = ref<InstanceType<typeof FileInput> | null>(null)

const handleFileChange = (file: File) => {
  selectedFile.value = file
}

// Helper function to format file size
const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const handleDetect = () => {
  if (selectedFile.value) {
    emit('detect', selectedFile.value)
    // Reset input after detection starts
    fileInputRef.value?.reset()
    // Clear selected file to prevent re-triggering
    selectedFile.value = null
  }
}
</script>

<template>
  <Card :title="title" :description="description">
    <div class="space-y-3">
      <!-- File Input -->
      <FileInput
        ref="fileInputRef"
        accept="image/*"
        label="选择图片文件"
        @change="handleFileChange"
      />

      <!-- File Name Display -->
      <div
        v-if="selectedFile"
        class="flex items-center justify-between bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg px-3 py-2"
      >
        <div class="flex-1 min-w-0">
          <div class="text-sm font-medium text-gray-700 dark:text-gray-300 truncate">
            {{ selectedFile.name }}
          </div>
          <div class="text-xs text-gray-500 dark:text-gray-400">
            {{ formatFileSize(selectedFile.size) }}
          </div>
        </div>
        <button
          @click="selectedFile = null"
          class="ml-2 text-sm text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
          type="button"
          aria-label="Clear selection"
        >
          清除
        </button>
      </div>

      <!-- Detect Button -->
      <button
        @click="handleDetect"
        :disabled="processing || !selectedFile"
        class="w-full bg-accent-600 hover:bg-accent-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium py-2 px-4 rounded-lg transition-colors"
      >
        {{ processing ? '检测中...' : '开始图片检测' }}
      </button>

      <!-- Error Message -->
      <div
        v-if="errorMessage"
        class="text-red-500 dark:text-red-400 text-sm"
      >
        {{ errorMessage }}
      </div>

      <!-- Preview Area -->
      <PreviewArea>
        <img
          v-if="resultUrl"
          :src="resultUrl"
          alt="图片检测结果"
          class="w-full h-full object-contain"
        />
        <span v-else class="text-gray-400 dark:text-gray-500">
          预览区域
        </span>
      </PreviewArea>

      <!-- Person Count Meta -->
      <div
        v-if="personCount !== undefined && personCount !== null"
        class="text-sm text-gray-700 dark:text-gray-300 bg-gray-50 dark:bg-gray-900/50 px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700"
      >
        检测到行人数：{{ personCount }}
      </div>
    </div>
  </Card>
</template>

<style scoped>
/* All styling via Tailwind utility classes */
</style>
