<script setup lang="ts">
interface Props {
  title?: string
  description?: string
}

withDefaults(defineProps<Props>(), {
  title: undefined,
  description: undefined,
})
</script>

<template>
  <section class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl shadow-lg p-4">
    <h2 v-if="title" class="text-xl font-semibold text-gray-900 dark:text-white">
      {{ title }}
    </h2>
    <div v-if="description" class="text-sm text-gray-600 dark:text-gray-400 mt-1.5 mb-3.5">
      {{ description }}
    </div>
    <slot />
  </section>
</template>

<style scoped>
/* All styling via Tailwind utility classes */
</style>
