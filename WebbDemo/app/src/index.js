import KerasJS from 'keras-js'

console.log('app is running')

const model = new KerasJS.Model({
  filepath: 'path/to/model.bin',
  gpu: true
})
