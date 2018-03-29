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

<style lang="less" scoped>
  .layer-outputs-container {

    @import '../variables';

    position: relative;

    .bg-line {
      position: absolute;
      z-index: 0;
      top: 0;
      left: 50%;
      background: whitesmoke;
      width: 15px;
      height: 100%;
    }

    .layer-output {
      position: relative;
      z-index: 1;
      margin: 30px 20px;
      background: whitesmoke;
      border-radius: 10px;
      padding: 20px;
      overflow-x: auto;

      .layer-output-heading {
        font-size: 1rem;
        color: #999999;
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
        font-size: 12px;

        span.layer-class {
          color: @color-green;
          font-size: 14px;
          font-weight: bold;
        }
      }

      .layer-output-canvas-container {
        display: inline-flex;
        flex-wrap: wrap;
        background: whitesmoke;

        canvas {
          border: 1px solid lightgray;
          margin: 1px;
        }
      }
    }
  }
</style>
