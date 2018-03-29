<template>

  <div class="container">

    <div class="row align-items-center">
      <div class="col">

        <div class="card">
          <div class="card-body">
            <h5>Hello, world!</h5>
            <p class="card-text">
                content
            </p>
          </div>
          <div class="card-body">
              <h5>Hello, world!</h5>
              <p class="card-text">
                  content
              </p>
            </div>
        </div>

      </div>

    </div>

  </div>

</template>

<script>
export default {
  name: 'HelloWorld',
  data () {
    return {
      msg: 'Welcome to Your Vue.js App'
    }
  },
  mounted: function () {
  this.$nextTick(function () {
    // Code that will run only after the
    // entire view has been rendered

    console.log('ready!')
  })
}
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h1, h2 {
  font-weight: normal;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
</style>
