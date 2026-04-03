<script setup lang="ts">
// Test dark mode toggle
const toggleDarkMode = () => {
  document.documentElement.classList.toggle('dark')
}
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors">
    <div class="max-w-4xl mx-auto py-12 px-4">
      <h1 class="text-3xl font-bold text-gray-900 dark:text-white mb-4">
        Tailwind CSS Test
      </h1>
      <p class="text-gray-600 dark:text-gray-400 mb-6">
        If you can see this, Tailwind is working!
      </p>
      <button
        @click="toggleDarkMode"
        class="btn-primary"
      >
        Toggle Dark Mode
      </button>
      <div class="mt-8 p-4 bg-white dark:bg-gray-800 rounded-lg shadow">
        <p class="text-sm text-gray-500 dark:text-gray-400">
          Background changes based on dark mode state
        </p>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* Component-specific styles can go here */
</style>
