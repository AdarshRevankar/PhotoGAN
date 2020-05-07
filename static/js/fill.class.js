import Point from "./point.model.js";

export default class Fill {
    constructor(canvas, point, color) {
        this.context = canvas.getContext("2d");

        this.imgData = this.context.getImageData(0, 0, this.context.canvas.width, this.context.canvas.height);

        const targetColor = this.getPixel(point)
        const fillColor = this.hexTorgb(color)
        this.fillStack = [];
        this.floodFill(point, targetColor, fillColor)
        this.fillColor();
    }

    floodFill(point, targetColor, fillColor) {
        if (this.colorMatch(targetColor, fillColor)) return

        const currentColor = this.getPixel(point)
        if (this.colorMatch(currentColor, targetColor)) {
            this.setPixel(point, fillColor)
        }
        if (this.colorMatch(targetColor, currentColor)) {
            this.setPixel(point,fillColor)
        this.fillStack.push([new Point(point.x + 1, point.y), targetColor, fillColor])
        this.fillStack.push([new Point(point.x - 1, point.y), targetColor, fillColor])
        this.fillStack.push([new Point(point.x, point.y + 1), targetColor, fillColor])
        this.fillStack.push([new Point(point.x + 1, point.y - 1), targetColor, fillColor])
        }
    }

    fillColor(){
        if(this.fillStack.length){
            let range=this.fillStack.length;
            for(let i=0;i<range;i++){
                this.floodFill(this.fillStack[i][0],this.fillStack[i][1],this.fillStack[i][2])
            }

            this.fillStack.splice(0,range);
            this.fillColor();
        }else{
            this.context.putImageData(this.imgData,0,0)
            this.fillStack=[]
        }
    }

    getPixel(point) {
        if (point.x < 0 || point.y < 0 || point.x >= this.imgData.width, point.y >= this.imgData.height) {
            return [-1, -1, -1, -1] //impossible color
        } else {
            const offset = (point.y * this.imgData.width + point.x) * 4;

           
            return [
                this.imgData.data[offset + 0],
                this.imgData.data[offset + 1],
                this.imgData.data[offset + 2],
                this.imgData.data[offset + 3]
            ]
        }

    }

    setPixel(point, fillColor) {
        const offset = (point.y * this.imgData.width + point.x) * 4;
        this.imgData.data[offset + 0] = fillColor[0]; //red
        this.imgData.data[offset + 1] = fillColor[1]; //green
        this.imgData.data[offset + 2] = fillColor[2]; //blue
        this.imgData.data[offset + 3] = fillColor[3]; //alpha
    }
    colorMatch(color1, color2) {
        return color1[0] === color2[0] && color1[1] === color2[1]
         && color1[2] === color1[2]&& color1[3] === color2[3];

    }
    hexTorgb(hex) {
        var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return [
            parseInt(result[1], 16),
            parseInt(result[2], 16),
            parseInt(result[3], 16),
            255
        ]
    }
}