<template>
  <div id="app">
    <latent-space-explorer/>
  </div>
</template>

<script>
import LatentSpaceExplorer from './components/LatentSpaceExplorer'


export default {
  name: 'App',
  components: {
    LatentSpaceExplorer
  }
}
</script>

<style lang="less">

@import './variables';

#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  /* margin-top: 60px; */

  height: 100vh;

}

.container {
  height:100%;
}

.row {
  height: 100%;
}
</style>
