import {
    TOOL_RECTANGLE,
    TOOL_CIRCLE,
    TOOL_TRIANGLE,
    TOOL_PENCIL,
    TOOL_BRUSH,
    TOOL_ERASER
} from './tool.js';
import Paint from './paint.class.js';

var paint = new Paint("canvas");
paint.activeTool = TOOL_PENCIL;
paint.brushSize = 4
paint.lineWidth = 1
paint.selectedColor = "#000000"
paint.init();
console.log("16", filename);
document.querySelectorAll("[data-command]").forEach(
    item => {
        item.addEventListener("click", e => {
            let command = item.getAttribute("data-command");
            if (command === 'undo') {
                paint.undoPaint()
            } else if (command === 'download') {
                var canvas = document.getElementById("canvas");
                var image = canvas.toDataURL("image/png", 1.0)
                    .replace("image/png", "image/active-stream")
                var link = document.createElement("a");
                link.download = document.getElementById("myName").value;
                link.href = image;
                link.click();
                filename = document.getElementById("myName").value;
            }
        })
    }
)
console.log("36", filename);

document.querySelectorAll("[data-tool]").forEach(
    item => {
        item.addEventListener("click", e => {
            document.querySelector("[data-tool].active").classList.toggle("active");
            item.classList.add("active")
            let selectedTool = item.getAttribute("data-tool")
            console.log(selectedTool)
            paint.activeTool = selectedTool;
            switch (selectedTool) {
                case TOOL_RECTANGLE:
                case TOOL_CIRCLE:
                case TOOL_TRIANGLE:
                case TOOL_PENCIL:
                    //activate shape linewidths group
                    document.querySelector(".group.for-shapes").style.display = "block"
                    //invisble brush linewidth group
                    document.querySelector(".group.for-brush").style.display = "none"
                    break;
                case TOOL_BRUSH:
                case TOOL_ERASER:
                    //activate brush linewidth group
                    document.querySelector(".group.for-brush").style.display = "block"
                    //invisble shape linewidth group
                    document.querySelector(".group.for-shapes").style.display = "none"
                    break;

                default:
                    //make invisible both 
                    document.querySelector(".group.for-shapes").style.display = "none"
                    document.querySelector(".group.for-brush").style.display = "none"
            }
        })
    }
)

document.querySelectorAll("[data-line-width]").forEach(
    item => {
        item.addEventListener("click", e => {
            document.querySelector("[data-line-width].active").classList.toggle("active");
            item.classList.add("active")
            let linewidth = item.getAttribute("data-line-width");
            paint.lineWidth = linewidth
        })
    }
)

document.querySelectorAll("[data-brush-width]").forEach(
    item => {
        item.addEventListener("click", e => {
            document.querySelector("[data-brush-width].active").classList.toggle("active");
            item.classList.add("active")
            let brushSize = item.getAttribute("data-brush-width");
            paint.brushSize = brushSize
        })
    }
)


document.querySelectorAll("[data-color]").forEach(
    item => {
        item.addEventListener("click", e => {
            document.querySelector("[data-color].active").classList.toggle("active");
            item.classList.add("active")

            let color = item.getAttribute("data-color");
            paint.selectedColor = color

        })
    }
)