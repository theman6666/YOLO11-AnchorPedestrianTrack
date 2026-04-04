<script setup lang="ts">
// Hero section component - static content, no props or state needed
</script>

<template>
  <section class="bg-gradient-to-br from-white to-gray-50 dark:from-gray-800 dark:to-gray-900 border border-gray-200 dark:border-gray-700 rounded-xl shadow-lg p-[22px] mb-4">
    <h1 class="text-3xl font-semibold text-accent-600 dark:text-accent-500 leading-relaxed">
      行人检测与跟踪系统
    </h1>
    <p class="text-sm text-gray-600 dark:text-gray-400 leading-relaxed mt-2.5">
      基于 YOLO11 + CBAM + ByteTrack，支持摄像头实时推理、图片检测和离线视频分析。
    </p>
    <div class="flex flex-wrap gap-2 mt-3.5">
      <span class="inline-flex items-center px-2.5 py-1.5 text-xs font-medium rounded-full border border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-300 bg-blue-50 dark:bg-blue-900/30">
        目标类别：行人
      </span>
      <span class="inline-flex items-center px-2.5 py-1.5 text-xs font-medium rounded-full border border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-300 bg-blue-50 dark:bg-blue-900/30">
        跟踪算法：ByteTrack
      </span>
      <span class="inline-flex items-center px-2.5 py-1.5 text-xs font-medium rounded-full border border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-300 bg-blue-50 dark:bg-blue-900/30">
        输出：人数统计 + FPS + 标注结果
      </span>
    </div>
  </section>
</template>

<style scoped>
/* All styling via Tailwind utility classes */
</style>
