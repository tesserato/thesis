var n = 10
var X = [...Array(n).keys()]
var W = Array(n)
var Wvis = Array(n).fill(true)
// var Colors = Plotly.d3.scale.linear().domain([0, n-1]).range(["blue", "red"])
var RED = "rgba(191, 63, 63, 0.2)"
var BLUE = "rgba(26,150,65,0.2)"

var RandomWaveLayout = {
  title:'Discrete Wave',
  font: {
    family: "Computer Modern",
    size: 18,
    color: "black"
  },
  yaxis:{
    range:[-1.1, 1.1]
  }
}
 
var p = 1.36
var f = 9.34

for (var j = 0; j < n; j++) {
  W[j] = Math.cos(p + 2.0 * Math.PI * f * j / n);
}

var RandomWaveData = X.map(
  (i) => {
    return {
      name:"",
      x:[i, i], 
      y:[0, W[i]], 
      type:'scatter',
      showlegend: false,
      marker: {color:"black"},
      // hovertemplate: "t = %{x}<br>W[%{x}] = %{y:.2f}"
    }
  }
)

Plotly.plot("RandomWave", RandomWaveData, RandomWaveLayout);


var x_vals = []
var x_text = []
for (let i = -20; i < 100; i+=.5) {
  x_vals.push(i * Math.PI)
  x_text.push("$ " + String(i) + " \\pi $")
}
var RegionsLayout = {
  title:'Isolines',
  font: {
    family: "Computer Modern",
    size: 18,
    color: "black"
  },
  xaxis: {
    range:[0 - 0.1, Math.PI + 0.1],
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

var RegionsData = []


for (var x = 0; x < n; x++) {
  var Xl = []
  var Yl = []
  if (W[x] >= 0) {
    for (var k = 0; k <= x + 1; k++) {
      Xl.push(2 * Math.PI * k + Math.acos(W[x]))                   ; Yl.push(0)
      Xl.push(2 * Math.PI * k + Math.acos(W[x]) - 2 * Math.PI * x) ; Yl.push(n)
      Xl.push(2 * Math.PI * k - Math.acos(W[x]) - 2 * Math.PI * x) ; Yl.push(n)
      Xl.push(2 * Math.PI * k - Math.acos(W[x]))                   ; Yl.push(0)
    }
    RegionsData.push(
      {
        name: x,
        x: Xl,
        y: Yl,
        type: "scatter",
        mode: 'lines',
        line:{width:0.5, color:"black"},
        fill: "tozeroy",
        fillcolor: W[x] >= 0 ? RED : BLUE,
        // showlegend: false,
      }
    )
  } else {
    for (var k = 0; k <= x + 1; k++) {
      Xl.push(2 * Math.PI * k + Math.acos(W[x]))                                 ; Yl.push(0)
      Xl.push(2 * Math.PI * k - 2 * Math.PI * x + Math.acos(W[x]))               ; Yl.push(n)
      Xl.push(2 * Math.PI * k - 2 * Math.PI * x - Math.acos(W[x]) + 2 * Math.PI) ; Yl.push(n)
      Xl.push(2 * Math.PI * k - Math.acos(W[x]) + 2 * Math.PI)                   ; Yl.push(0)
    }
    RegionsData.push(
      {
        name: x,
        x: Xl,
        y: Yl,
        type: "scatter",
        mode: 'lines',
        line:{width:0.5, color:"black"},
        fill: "tozeroy",
        fillcolor: W[x] >= 0 ? RED : BLUE,
        // showlegend: false,
      }
    )

  }
  // Xl = []
  // Yl = []
  // for (var k = 0; k <= x + 1; k++) {
  // Xl.push(2 * Math.PI * k + Math.PI / 2)                     ; Yl.push(0)
  // Xl.push(2 * Math.PI * k - 2 * Math.PI * x + Math.PI / 2)   ; Yl.push(n)
  // Xl.push(2 * Math.PI * k - 2 * Math.PI * x + 1.5 * Math.PI) ; Yl.push(n)
  // Xl.push(2 * Math.PI * k + 1.5 * Math.PI)                   ; Yl.push(0)
  // }
  // RegionsData.push(
  //   {
  //     name: "",
  //     x: Xl,
  //     y: Yl,
  //     type: "scatter",
  //     mode: 'lines', 
  //     line:{width:0.5, color:"black"},
  //     fill: "tozeroy",
  //     fillcolor: W[x] >= 0 ? "rgba(26,150,65,0.2)" : "rgba(191, 63, 63, 0.2)",
  //     showlegend: false,
  //   }
  // )
}

Plotly.plot("Isolines", RegionsData, RegionsLayout);

var dt = [
  {
    name: "Unique Space",
    x: [0, 0, Math.PI, Math.PI, 0],
    y: [0, n/2, n/2, 0, 0],
    type: "scatter",
    mode: "lines",
    line:{width:2, color:"black"},
    showlegend: false,
  },
  {
    name: "Max Frequency",
    x: [0, Math.PI],
    y: [f, f],
    type: "scatter",
    mode: "lines",
    line:{width:1, color:"red"},
    showlegend: false,
  },
  {
    name: "Max Phase",
    x: [p, p],
    y: [0, n/2],
    type: "scatter",
    mode: "lines",
    line:{width:1, color:"red"},
    showlegend: false,
  }
]

Plotly.plot("Isolines", dt)

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