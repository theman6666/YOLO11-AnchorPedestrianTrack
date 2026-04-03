<script setup lang="ts">
import { Camera, Image, Video, Settings, Info } from 'lucide-vue-next'
import ApiTest from '@/components/ApiTest.vue'

const toggleDarkMode = () => {
  document.documentElement.classList.toggle('dark')
}
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors">
    <div class="max-w-4xl mx-auto py-12 px-4">
      <h1 class="text-3xl font-bold text-gray-900 dark:text-white mb-4">
        Tailwind CSS + Lucide Icons Test
      </h1>
      <p class="text-gray-600 dark:text-gray-400 mb-6">
        Testing Tailwind CSS and Lucide icon integration
      </p>

      <!-- Icon test section -->
      <div class="flex gap-4 mb-6">
        <div class="flex items-center gap-2 text-gray-700 dark:text-gray-300">
          <Camera :size="24" />
          <span>Camera</span>
        </div>
        <div class="flex items-center gap-2 text-gray-700 dark:text-gray-300">
          <Image :size="24" />
          <span>Image</span>
        </div>
        <div class="flex items-center gap-2 text-gray-700 dark:text-gray-300">
          <Video :size="24" />
          <span>Video</span>
        </div>
        <div class="flex items-center gap-2 text-gray-700 dark:text-gray-300">
          <Settings :size="24" />
          <span>Settings</span>
        </div>
        <div class="flex items-center gap-2 text-gray-700 dark:text-gray-300">
          <Info :size="24" />
          <span>Info</span>
        </div>
      </div>

      <button
        @click="toggleDarkMode"
        class="btn-primary"
      >
        Toggle Dark Mode
      </button>

      <div class="mt-8 p-4 bg-white dark:bg-gray-800 rounded-lg shadow">
        <p class="text-sm text-gray-500 dark:text-gray-400">
          If icons render above, Lucide integration is working!
        </p>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* Component-specific styles can go here */
</style>
