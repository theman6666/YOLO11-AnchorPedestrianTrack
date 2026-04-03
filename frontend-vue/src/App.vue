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
        YOLO11 Frontend - Project Setup Verification
      </h1>
      <p class="text-gray-600 dark:text-gray-400 mb-6">
        Testing Tailwind CSS, Lucide icons, and Axios API integration
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
        class="px-4 py-2 bg-accent-600 text-white rounded-lg hover:bg-accent-700 transition-colors mb-6"
      >
        Toggle Dark Mode
      </button>

      <!-- API proxy test component -->
      <ApiTest />

      <!-- Status info -->
      <div class="mt-8 p-4 bg-white dark:bg-gray-800 rounded-lg shadow">
        <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
          Setup Status
        </h3>
        <ul class="text-xs text-gray-600 dark:text-gray-400 space-y-1">
          <li>Tailwind CSS configured</li>
          <li>Dark mode working</li>
          <li>Lucide icons rendering</li>
          <li>Axios client configured</li>
          <li>Vite proxy configured</li>
        </ul>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* Component-specific styles can go here */
</style>
