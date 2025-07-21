document.addEventListener('DOMContentLoaded', function () {
  let canvas = document.getElementById("canvas");
  let ctx = canvas.getContext("2d");

  ctx.lineWidth = 20;
  ctx.lineCap = "round";

  let drawing = false;

  canvas.addEventListener("mousedown", () => drawing = true);
  canvas.addEventListener("mouseup", () => drawing = false);
  canvas.addEventListener("mouseout", () => drawing = false);
  canvas.addEventListener("mousemove", draw);

  function draw(e) {
    if (!drawing) return;
    ctx.strokeStyle = "black";
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
    ctx.lineTo(e.offsetX + 0.1, e.offsetY + 0.1);
    ctx.stroke();
  }

  window.clearCanvas = function () {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  window.submitCanvas = function () {
    let imgData = canvas.toDataURL("image/png");
    fetch('/', {
      method: 'POST',
      body: JSON.stringify({ image: imgData }),
      headers: { 'Content-Type': 'application/json' }
    })
    .then(res => res.json())
    .then(data => {
      document.getElementById("prediction-result").innerText = "Predicted Digit: " + data.prediction;
    });
  }
});
