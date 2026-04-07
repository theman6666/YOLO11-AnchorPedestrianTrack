<script setup lang="ts">
import { ref } from 'vue'
import { Upload } from 'lucide-vue-next'
import PreviewArea from '@/components/ui/PreviewArea.vue'

interface Props {
  accept?: string
  label?: string
  disabled?: boolean
}

withDefaults(defineProps<Props>(), {
  accept: '*',
  label: '选择文件',
  disabled: false,
})

const emit = defineEmits<{
  change: [file: File]
}>()

const inputRef = ref<HTMLInputElement | null>(null)
const handleFileChange = (event: Event) => {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (file) {
    emit('change', file)
  }
}

// Expose reset method for parent components
defineExpose({
  reset: () => {
    if (inputRef.value) {
      inputRef.value.value = ''
    }
  }
})
</script>

<template>
  <div class="space-y-2">
    <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">
      {{ label }}
    </label>

    <!-- Drag-drop visual overlay using PreviewArea -->
    <div class="relative group cursor-pointer">
      <PreviewArea minHeight="120px">
        <div class="flex flex-col items-center justify-center space-y-2">
          <!-- Upload icon (use lucide-vue-next Upload icon) -->
          <Upload
            :size="32"
            class="text-gray-400 dark:text-gray-500 group-hover:text-accent-600 dark:group-hover:text-accent-500 transition-colors"
          />
          <span class="text-sm text-gray-500 dark:text-gray-400 group-hover:text-gray-700 dark:group-hover:text-gray-300 transition-colors">
            点击或拖拽文件到此处
          </span>
        </div>
      </PreviewArea>

      <!-- Hidden file input -->
      <input
        ref="inputRef"
        type="file"
        :accept="accept"
        :disabled="disabled"
        @change="handleFileChange"
        class="absolute inset-0 opacity-0 cursor-pointer"
        :class="{ 'pointer-events-none': disabled }"
      />
    </div>
  </div>
</template>

<style scoped>
/* All styling via Tailwind utility classes */
</style>
