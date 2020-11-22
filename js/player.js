class Player {

    constructor(container) {
        this.container = container
        this.global_frac = 0.0
        this.container = document.getElementById(container)
        this.progress = null;
        this.mat = [[]]

        this.player = this.container.querySelector('audio')
        this.demo_img = this.container.querySelector('.underlay > img')
        this.underlay = this.container.querySelector('.underlay')
        this.playpause = this.container.querySelector(".playpause");
        this.play_img = this.container.querySelector('.play-img')
        this.pause_img = this.container.querySelector('.pause-img')
        this.canvas = this.container.querySelector('.response-canvas')
        this.response_container = this.container.querySelector('.response')
        this.context = this.canvas.getContext('2d');

        var togglePlayPause = () => {
            if (this.player.networkState !== 1) {
                return
            }
            if (this.player.paused || this.player.ended) {
                this.play()
            } else {
                this.pause()
            }
        }

        this.updateLoop = (timestamp) => {
            this.global_frac = this.player.currentTime / this.player.duration
            this.redraw()
            this.progress = window.requestAnimationFrame(this.updateLoop)
        }

        this.playpause.disabled = true
        this.player.onplay = this.updateLoop
        this.player.onpause = () => {
            window.cancelAnimationFrame(this.progress)
            this.global_frac = this.player.currentTime / this.player.duration
            this.redraw();  // final redraw
        }
        this.player.onended = () => {this.pause()}
        this.playpause.onclick = togglePlayPause;
    }

    loadAndRedraw(audio_fname, img_fname) {
        // stop on-going animation
        this.pause()
        window.cancelAnimationFrame(this.progress)

        // remove image, disable play button and player
        this.demo_img.style.display = 'none'
        this.playpause.disabled = true
        this.redrawPlayer(true)

        // reload player
        this.player.querySelector('#src1').setAttribute("src", audio_fname + '.ogg')
        this.player.querySelector('#src2').setAttribute("src", audio_fname + '.wav')
        this.player.load()  // load() function of <audio> element

        this.demo_img.setAttribute("src", img_fname)

        // fetch data
        fetch(img_fname)
          .then(response => response.arrayBuffer())
          .then(text => {
            this.mat = this.parse(text);
            this.playpause.disabled = false;
            this.demo_img.style.display = 'inline'

            this.global_frac = 0.0
            this.redraw();
        })

    }

    parse(buffer) {
        var img = UPNG.decode(buffer)
        var dat = UPNG.toRGBA8(img)[0]
        var view = new DataView(dat)
        var data = new Array(img.width).fill(0).map(() => new Array(img.height).fill(0));

        var min =100
        var max = -100
        var idx = 0
        for (let i=0; i < img.height*img.width*4; i+=4) {
            var rgba = [view.getUint8(i, 1) / 255, view.getUint8(i + 1, 1) / 255, view.getUint8(i + 2, 1) / 255, view.getUint8(i + 3, 1) / 255]
            var norm = Math.pow(Math.pow(rgba[0], 2) + Math.pow(rgba[1], 2) + Math.pow(rgba[2], 2), 0.5)
            data[idx % img.width][img.height - Math.floor(idx / img.width) - 1] = norm

            idx += 1
            min = Math.min(min, norm)
            max = Math.max(max, norm)
        }
        for (let i = 0; i < data.length; i++) {
            for (let j = 0; j < data[i].length; j++) {
                data[i][j] = Math.pow((data[i][j] - min) / (max - min), 1.5)
            }
        }
        var data3 = new Array(img.width).fill(0).map(() => new Array(img.height).fill(0));
        for (let i = 0; i < data.length; i++) {
            for (let j = 0; j < data[i].length; j++) {
                if (i == 0 || i == (data.length - 1)) {
                    data3[i][j] = data[i][j]
                } else{
                    data3[i][j] = 0.33*(data[i - 1][j]) + 0.33*(data[i][j]) + 0.33*(data[i + 1][j])
                    // data3[i][j] = 0.00*(data[i - 1][j]) + 1.00*(data[i][j]) + 0.00*(data[i + 1][j])
                }
            }
        }

        var scale = 5
        var data2 = new Array(scale*img.width).fill(0).map(() => new Array(img.height).fill(0));
        for (let j = 0; j < data[0].length; j++) {
            for (let i = 0; i < data.length - 1; i++) {
                for (let k = 0; k < scale; k++) {
                    data2[scale*i + k][j] = (1.0 - (k/scale))*data3[i][j] + (k / scale)*data3[i + 1][j]
                }
            }
        }
        return data2
    }

    play() {
        this.player.play();
        this.play_img.style.display = 'none'
        this.pause_img.style.display = 'block'
    }

    pause() {
        this.player.pause();
        this.pause_img.style.display = 'none'
        this.play_img.style.display = 'block'
    }

    redraw() {
        // set cropping position of image
        this.cropImage(this.global_frac)

        // draw spectrogram in player
        this.redrawPlayer(this.global_frac)
    }

    redrawPlayer(global_frac, greyedOut = false) {
        // only redraw when there is information to redraw
        if (this.mat.length > 1) {
            this.canvas.width = window.devicePixelRatio*this.response_container.offsetWidth;
            this.canvas.height = window.devicePixelRatio*this.response_container.offsetHeight;

            this.context.clearRect(0, 0, this.canvas.width, this.canvas.height)
            this.canvas.style.width = (this.canvas.width / window.devicePixelRatio).toString() + "px";
            this.canvas.style.height = (this.canvas.height / window.devicePixelRatio).toString() + "px";

            var f = global_frac * this.mat.length
            var tstep = Math.min(Math.floor(f), this.mat.length - 2)
            var heights = this.mat[tstep]
            var bar_width = (this.canvas.width / heights.length) - 1

            for (let k = 0; k < heights.length - 1; k++) {
                var height = Math.max(Math.round((heights[k])*this.canvas.height), 3)
                this.context.fillStyle = greyedOut? '#c8c8c8' : '#696f7b';
                this.context.fillRect(k*(bar_width + 1), (this.canvas.height - height) / 2, bar_width, height);
            }
        }
    }

    cropImage(global_frac) {
        var imageRescalingFactor = this.demo_img.height / this.demo_img.naturalHeight
        var imageFullWitdh = imageRescalingFactor * this.demo_img.naturalWidth
        var objectPositionStart = this.underlay.offsetWidth / 2.
        var objectPositionEnd = -imageFullWitdh + this.underlay.offsetWidth / 2.

        this.demo_img.style.objectPosition = (global_frac * (objectPositionEnd - objectPositionStart) + objectPositionStart).toString() + 'px 0%'
    }
}