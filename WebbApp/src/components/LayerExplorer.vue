<template>
  <div class="layer-outputs-container" v-if="!modelLoading">
    <div class="bg-line"></div>
    <div
      v-for="(layerOutput, layerIndex)  in layerOutputImages"
      :key="`intermediate-output-${layerIndex}`"
      class="layer-output"
    >
      <div class="layer-output-heading">
        <span class="layer-class">{{ layerOutput.layerClass }}</span>
        <span> {{ layerDisplayConfig[layerOutput.name].heading }}</span>
      </div>
      <div class="layer-output-canvas-container">
        <canvas v-for="(image, index) in layerOutput.images"
          :key="`intermediate-output-${layerIndex}-${index}`"
          :id="`intermediate-output-${layerIndex}-${index}`"
          :width="image.width"
          :height="image.height"
          style="display:none;"
        ></canvas>
        <canvas v-for="(image, index) in layerOutput.images"
          :key="`intermediate-output-${layerIndex}-${index}-scaled`"
          :id="`intermediate-output-${layerIndex}-${index}-scaled`"
          :width="layerDisplayConfig[layerOutput.name].scalingFactor * image.width"
          :height="layerDisplayConfig[layerOutput.name].scalingFactor * image.height"
        ></canvas>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  props: ['modelLoading', 'layerOutputImages', 'layerDisplayConfig'],
  data(){
    return {

    }
  }
}
</script>
