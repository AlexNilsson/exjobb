import Vue from 'vue'

import BootstrapVue from 'bootstrap-vue'
import 'bootstrap/dist/css/bootstrap.css'
import 'bootstrap-vue/dist/bootstrap-vue.css'

// https://github.com/euvl/vue-js-toggle-button
import ToggleButton from 'vue-js-toggle-button'

import App from './App'

Vue.use(BootstrapVue)
Vue.use(ToggleButton)

Vue.config.productionTip = false

/* eslint-disable no-new */
new Vue({
  el: '#app',
  render: h => h(App)
})
