import Point from './point.model.js'
import {
    TOOL_LINE,
    TOOL_RECTANGLE,
    TOOL_CIRCLE,
    TOOL_TRIANGLE,
    TOOL_PAINT_BUCKET,
    TOOL_PENCIL,
    TOOL_BRUSH,
    TOOL_ERASER
} from './tool.js';
import {
    getMouseCoordsOnCanvas,
    FindDistance
} from './utility.js'
import Fill from './fill.class.js';
export default class Paint {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.context = canvas.getContext("2d");
        this.undoStack = [];
        this.undoLimit = 3;
    }
    set activeTool(tool) {
        this.tool = tool;
    }
    set lineWidth(linewidth) {
        this._lineWidth = linewidth;
        this.context.lineWidth = this._lineWidth //set width to global
    }
    set brushSize(brushsize) {
        this._brushSize = brushsize

    }
    set selectedColor(color) {
        this.color = color;

        this.context.strokeStyle = this.color //set color to global
    }
    init() {
        this.canvas.onmousedown = e => this.onMouseDown(e);
    }
    onMouseDown(e) {
        this.saveData = this.context.getImageData(0, 0, this.canvas.clientWidth, this.canvas.clientHeight)
        if(this.undoStack.length>=this.undoLimit) {this.undoStack.shift();}//remove first element of array.
        console.log(this.undoStack)
        this.undoStack.push(this.saveData)

        this.canvas.onmousemove = e => this.onMouseMove(e);
        document.onmouseup = e => this.onMouseUp(e);
        this.startPo = getMouseCoordsOnCanvas(e, this.canvas);

        if (this.tool == TOOL_PENCIL || this.tool == TOOL_BRUSH) {
            this.context.beginPath()
            this.context.moveTo(this.startPo.x, this.startPo.y)
        } else if (this.tool == TOOL_PAINT_BUCKET) {
            //FILL THE COLOR
            new Fill(this.canvas, this.startPo, this.color);
        } else if (this.tool == TOOL_ERASER) {
            this.context.clearRect(this.startPo.x, this.startPo.y,
                this._brushSize, this._brushSize)
        }

    }
    onMouseMove(e) {
        this.currentPos = getMouseCoordsOnCanvas(e, this.canvas)

        

        switch (this.tool) {
            case TOOL_LINE:
            case TOOL_RECTANGLE:
            case TOOL_CIRCLE:
            case TOOL_TRIANGLE:
                this.drawShape();
                break;
            case TOOL_PENCIL:
                this.drawFreeLine(this._lineWidth);
                break;
            case TOOL_BRUSH:
                this.drawFreeLine(this._brushSize)
                break;
            case TOOL_ERASER:
                this.context.clearRect(this.currentPos.x, this.currentPos.y,
                    this._brushSize, this._brushSize)
            default:
                break;
        }
    }
    onMouseUp(e) {
        this.canvas.onmousemove = null;
        document.onmouse = null;
    }
    drawShape() {
        this.context.putImageData(this.saveData, 0, 0)
        this.context.beginPath();


        if (this.tool == TOOL_LINE) {
            this.context.moveTo(this.startPo.x, this.startPo.y)
            this.context.lineTo(this.currentPos.x, this.currentPos.y);
        } else if (this.tool == TOOL_RECTANGLE) {
            this.context.rect(this.startPo.x, this.startPo.y, this.currentPos.x - this.startPo.x, this.currentPos.y - this.startPo.y)
        } else if (this.tool == TOOL_CIRCLE) {
            let distance = FindDistance(this.startPo, this.currentPos);
            this.context.arc(this.startPo.x, this.startPo.y, distance, 0, 2 * Math.PI, false)
        } else if (this.tool == TOOL_TRIANGLE) {
            this.context.moveTo(this.startPo.x + (this.currentPos.x - this.startPo.x) / 2, this.startPo.y);
            this.context.lineTo(this.startPo.x, this.currentPos.y);
            this.context.lineTo(this.currentPos.x, this.currentPos.y);
            this.context.closePath();
        }
        this.context.stroke();

    }
    drawFreeLine(lineWidth) {
        this.context.lineWidth = lineWidth;
        this.context.lineTo(this.currentPos.x, this.currentPos.y);
        this.context.stroke();
    }

    undoPaint(){
        if(this.undoStack.length>0){
            this.context.putImageData(this.undoStack[this.undoStack.length-1],0,0);
            this.undoStack.pop();//also remove from stack
        
        }else{
            alert('No Undo Available;')
        }
    }
}