import { resolve } from 'path'

export default {
    root: '.',
    base: '/static/dist/',
    build: {
        outDir: '../static/dist',
        emptyOutDir: true,
        rollupOptions: {
            input: {
                main: resolve(__dirname, 'main.js')
            },
            output: {
                entryFileNames: '[name].js',       // disables content hashing
                chunkFileNames: '[name].js',
                assetFileNames: '[name][extname]'
            }
        }
    }
}
