<script setup lang="ts">
// PanelGrid component - responsive 3-panel grid layout wrapper
// No props needed - layout-only component with slot for content
</script>

<template>
  <div class="grid grid-cols-[repeat(auto-fit,minmax(340px,1fr))] gap-[14px]">
    <slot />
  </div>
</template>

<style scoped>
/* All styling via Tailwind utility classes */
</style>
