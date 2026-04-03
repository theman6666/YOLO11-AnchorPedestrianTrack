<script setup lang="ts">
interface Props {
  minHeight?: string
}

withDefaults(defineProps<Props>(), {
  minHeight: '230px',
})
</script>

<template>
  <div
    class="min-h-[230px] border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-xl bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-850 flex items-center justify-center overflow-hidden text-gray-500 dark:text-gray-400 text-sm p-2"
    :style="{ minHeight: minHeight || '230px' }"
  >
    <slot>
      <span class="text-gray-400 dark:text-gray-500">预览区域</span>
    </slot>
  </div>
</template>

<style scoped>
/* All styling via Tailwind utility classes */
</style>
