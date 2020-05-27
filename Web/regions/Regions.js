var n = 10
var X = [...Array(n).keys()]
var W = Array(n)
var Wvis = Array(n).fill(true)
var Colors = Plotly.d3.scale.linear().domain([0, n-1]).range(["blue", "red"])

var RandomWaveData
var RandomWaveLayout

var RegionsData
var RegionsLayout

function RandomWave() {
  W.fill(0.0);
  for (var i=0; i < 10; i++) {
    var f = Math.random() * n;
    var p = Math.random() * 2.0 * Math.PI;
    for (var j = 0; j < n; j++) {
      W[j] += Math.cos(p + 2.0 * Math.PI * f * X[j] / n);
    }
  }
  
  var maxW = Math.max(...W.map(i => Math.abs(i)))
  W = W.map(i => i / maxW)

  RandomWaveData = X.map(
    (i) => {
      return {
        name:"",
        x:[i, i], 
        y:[0, W[i]], 
        type:'scatter',
        showlegend: false,
        marker: {color:"black"},
        hovertemplate: "t = %{x}<br>W[%{x}] = %{y:.2f}"
      }
    }
  )
}
RandomWave();

RandomWaveLayout = {
  title:'Discrete Wave',
  font: {
    family: "Computer Modern",
    size: 18,
    color: "black"
  },
  yaxis:{
    range:[-1.1, 1.1]
  }
};
Plotly.plot("RandomWave", RandomWaveData, RandomWaveLayout);

RegionsData = []
var Xl = []
var Yl = []

for (var t = 0; t < n; t++) {
  Xl = []
  Yl = []
  for (var k = 0; k <= t + 1; k++) {
    Xl.push(2 * Math.PI * k - Math.PI / 2)                   ; Yl.push(0)
    Xl.push(2 * Math.PI * k - 2 * Math.PI * t - Math.PI / 2) ; Yl.push(n)
    Xl.push(2 * Math.PI * k - 2 * Math.PI * t + Math.PI / 2) ; Yl.push(n)
    Xl.push(2 * Math.PI * k + Math.PI / 2)                   ; Yl.push(0)
  }
  RegionsData.push(
    {
      name: "",
      x: Xl,
      y: Yl,
      type: "scatter",
      mode: "none",
      fill: "tozeroy",
      fillcolor: W[t] >= 0 ? "rgba(191, 63, 63, 0.2)" : "rgba(26,150,65,0.2)",
      showlegend: false,
    }
  )

  Xl = []
  Yl = []
  for (var k = 0; k <= t + 1; k++) {
  Xl.push(2 * Math.PI * k + Math.PI / 2)                     ; Yl.push(0)
  Xl.push(2 * Math.PI * k - 2 * Math.PI * t + Math.PI / 2)   ; Yl.push(n)
  Xl.push(2 * Math.PI * k - 2 * Math.PI * t + 1.5 * Math.PI) ; Yl.push(n)
  Xl.push(2 * Math.PI * k + 1.5 * Math.PI)                   ; Yl.push(0)
  }
  RegionsData.push(
    {
      name: "",
      x: Xl,
      y: Yl,
      type: "scatter",
      mode: "none",
      fill: "tozeroy",
      fillcolor: W[t] >= 0 ? "rgba(26,150,65,0.2)" : "rgba(191, 63, 63, 0.2)",
      showlegend: false,
    }
  )
}

var x_vals = []
var x_text = []
for (let i = -20; i < 100; i++) {
  x_vals.push(i * Math.PI)
  x_text.push("$ " + String(i) + " \\pi $")
}

RegionsLayout = {
  title:'Isolines',
  font: {
    family: "Computer Modern",
    size: 18,
    color: "black"
  },
  xaxis: {
    range:[0 - 0.1, 2 * Math.PI + 0.1],
    tickvals: x_vals,
    ticktext: x_text,
    // zerolinecolor: "gray",
    // zerolinewidth: 6,
    // gridcolor: "black",
    gridwidth: 2
  },
  yaxis: {
    range:[0 - 0.5, n/2],
    scaleanchor:"x",
    // zerolinecolor: "gray",
    // zerolinewidth: 6,
    // gridcolor: "black",
    gridwidth: 2
  }
}

Plotly.plot("Isolines", RegionsData, RegionsLayout);

function plot_unique_space(){
  var dt = [{
    name: "Unique Space",
    x: [0, 0, Math.PI, Math.PI, 0],
    y: [0, n/2, n/2, 0, 0],
    type: "scatter",
    mode: "lines",
    // fill: "tozeroy",
    // fillcolor: W[t] >= 0 ? "rgba(26,150,65,0.2)" : "rgba(191, 63, 63, 0.2)",
    showlegend: false,
  }]
  Plotly.plot("Isolines", dt)
}

plot_unique_space()

function Update() {

  RandomWave();

  for (let i = 0; i < n; i++) {

      RegionsData[2 * i].fillcolor = W[i] >= 0 ? "rgba(191, 63, 63, 0.2)" : "rgba(26,150,65,0.2)"
      // console.log("even")
      RegionsData[2 * i + 1].fillcolor = W[i] >= 0 ? "rgba(26,150,65,0.2)" : "rgba(191, 63, 63, 0.2)"
      // console.log("odd")
  }

  Plotly.react("RandomWave", RandomWaveData, RandomWaveLayout )
  Plotly.react("Isolines", RegionsData, RegionsLayout)
  // plot_unique_space()
}


var RandomWaveDiv = document.getElementById("RandomWave");

RandomWaveDiv.on('plotly_click', function(data){
  var t = data.points[0].curveNumber;
  var RandomWaveUpdate
  var IsolinesUpdate
  if (Wvis[t]) {
    RandomWaveUpdate = {'marker':{color: "gray"}};
    IsolinesUpdate = {"visible": false}
    Wvis[t] = false;
  }else{
    RandomWaveUpdate = {'marker':{color: "black"}};
    IsolinesUpdate = {"visible": true}
    Wvis[t] = true;
  }


  Plotly.restyle("RandomWave", RandomWaveUpdate, [t])
  Plotly.restyle("Isolines", IsolinesUpdate, [t*2, t*2+1])
});

document.getElementById("UpdateBtn").onclick = function() {Update()};

document.getElementById("ShowBtn").onclick = function() {
  Wvis.fill(true)
  var RandomWaveUpdate = {'marker':{color: "black"}};
  var IsolinesUpdate = {"visible": true}

  Plotly.restyle("RandomWave", RandomWaveUpdate)
  Plotly.restyle("Isolines", IsolinesUpdate)
};

document.getElementById("HideBtn").onclick = function() {
  Wvis.fill(false)
  var RandomWaveUpdate = {'marker':{color: "gray"}}
  var IsolinesUpdate = {"visible": false}
  var traces = [...Array(2 * n).keys()]
  Plotly.restyle("RandomWave", RandomWaveUpdate, traces)
  Plotly.restyle("Isolines", IsolinesUpdate, traces)
};