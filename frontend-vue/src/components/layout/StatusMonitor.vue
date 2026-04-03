<script setup lang="ts">
interface Props {
  message: string
  isOk?: boolean
}

withDefaults(defineProps<Props>(), {
  isOk: false,
})
</script>

<template>
  <div
    class="mt-3.5 border rounded-xl px-3 py-2.5 text-sm transition-colors"
    :class="
      isOk
        ? 'border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-300'
        : 'border-yellow-200 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300'
    "
  >
    {{ message }}
  </div>
</template>

<style scoped>
/* All styling via Tailwind utility classes */
</style>
