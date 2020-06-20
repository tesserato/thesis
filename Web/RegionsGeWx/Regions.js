var n = 10
var X = [...Array(n).keys()]
var W = Array(n).fill(0.0)
var Wvis = Array(n).fill(true)
// var Colors = Plotly.d3.scale.linear().domain([0, n-1]).range(["blue", "red"])

var number_of_random_sinusoids = 2
var F = Array(number_of_random_sinusoids)
var A = Array(number_of_random_sinusoids)
var P = Array(number_of_random_sinusoids)

/////////////////////////////////////////////
/////////////////////////////////////////////
function RandomWave() {
  for (var i=0; i < number_of_random_sinusoids; i++) {
    A[i] = Math.random()
    F[i] = Math.random() * n / 2;
    P[i] = Math.random() * Math.PI;
    for (var j = 0; j < n; j++) {
      W[j] +=  Math.cos(P[i] + 2.0 * Math.PI * F[i] * j / n);
    }
  }
  
  var maxW = Math.max(...W.map(i => Math.abs(i))) //+ 0.001
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
/////////////////////////////////////////////
/////////////////////////////////////////////

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
  var Xr_e = []; var Xr_o = []
  var Yr_e = []; var Yr_o = []
  var Xl = []//; var Xl = []
  var Yl = []//; var Yl = []
  
  var RED  = `rgba(191, 63, 63, 0.2)`
  var BLUE = `rgba(26, 150, 65, 0.2)`
  var RED  = `rgba(191, 63, 63, ${Math.abs(W[x])/4})`
  var BLUE = `rgba(26, 150, 65, ${Math.abs(W[x])/4})`

  for (var k = 0; k <= x + 1; k++) {
    if (W[x] >= 0) {
      // even
      Xr_e.push(2 * Math.PI * k + Math.acos(W[x]))                   ; Yr_e.push(0)
      Xr_e.push(2 * Math.PI * k + Math.acos(W[x]) - 2 * Math.PI * x) ; Yr_e.push(n)
      Xr_e.push(2 * Math.PI * k - Math.acos(W[x]) - 2 * Math.PI * x) ; Yr_e.push(n)
      Xr_e.push(2 * Math.PI * k - Math.acos(W[x]))                   ; Yr_e.push(0)
      // odd
      Xr_o.push(2 * Math.PI * (k + .5) + Math.acos(W[x]))                   ; Yr_o.push(0)
      Xr_o.push(2 * Math.PI * (k + .5) + Math.acos(W[x]) - 2 * Math.PI * x) ; Yr_o.push(n)
      Xr_o.push(2 * Math.PI * (k + .5) - Math.acos(W[x]) - 2 * Math.PI * x) ; Yr_o.push(n)
      Xr_o.push(2 * Math.PI * (k + .5) - Math.acos(W[x]))                   ; Yr_o.push(0)

      Xl.push(2 * Math.PI * k)                         ; Yl.push(0)
      Xl.push(Math.PI * (-2 * n * x + n * (2 * k)) / n); Yl.push(n)
      Xl.push(NaN) ; Yl.push(NaN)
      Xl.push(2 * Math.PI * k + Math.PI)                   ; Yl.push(0)
      Xl.push(Math.PI * (-2 * n * x + n * (2 * k + 1)) / n); Yl.push(n)
      Xl.push(NaN) ; Yl.push(NaN)
    } else {
      // even
      Xr_e.push(2 * Math.PI * (k - .5) + Math.acos(W[x]))                                 ; Yr_e.push(0)
      Xr_e.push(2 * Math.PI * (k - .5) - 2 * Math.PI * x + Math.acos(W[x]))               ; Yr_e.push(n)
      Xr_e.push(2 * Math.PI * (k - .5) - 2 * Math.PI * x - Math.acos(W[x]) + 2 * Math.PI) ; Yr_e.push(n)
      Xr_e.push(2 * Math.PI * (k - .5) - Math.acos(W[x]) + 2 * Math.PI)                   ; Yr_e.push(0)
      // odd
      Xr_o.push(2 * Math.PI * k + Math.acos(W[x]))                                 ; Yr_o.push(0)
      Xr_o.push(2 * Math.PI * k - 2 * Math.PI * x + Math.acos(W[x]))               ; Yr_o.push(n)
      Xr_o.push(2 * Math.PI * k - 2 * Math.PI * x - Math.acos(W[x]) + 2 * Math.PI) ; Yr_o.push(n)
      Xr_o.push(2 * Math.PI * k - Math.acos(W[x]) + 2 * Math.PI)                   ; Yr_o.push(0)

      Xl.push(2 * Math.PI * k)                         ; Yl.push(0)
      Xl.push(Math.PI * (-2 * n * x + n * (2 * k)) / n); Yl.push(n)
      Xl.push(NaN) ; Yl.push(NaN)
      Xl.push(2 * Math.PI * k + Math.PI)                   ; Yl.push(0)
      Xl.push(Math.PI * (-2 * n * x + n * (2 * k + 1)) / n); Yl.push(n)
      Xl.push(NaN) ; Yl.push(NaN)
    }
  }

    RegionsData.push(
    {
      name: x,
      legendgroup: x,
      x: Xr_e,
      y: Yr_e,
      type: "scatter",
      mode: 'lines',
      line:{width:0.5, color:"black"},
      fill: "tozeroy",
      fillcolor: W[x] >= 0 ? RED : BLUE,
      // showlegend: false,
    },
    {
      name: x,
      legendgroup: x,
      x: Xr_o,
      y: Yr_o,
      type: "scatter",
      mode: 'lines',
      line:{width:0.5, color:"black"},
      fill: "tozeroy",
      fillcolor: W[x] >= 0 ? BLUE : RED,
      // showlegend: false,
    },
    {
      name: x,
      legendgroup: x,
      x: Xl,
      y: Yl,
      type: "scatter",
      mode: 'lines',
      line:{width:0.5, color:"black"},
      showlegend: false,
    }
  )
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
    x: P,
    y: F,
    type: "scatter",
    mode: "markers",
    // line:{width:1, color:"red"},
    showlegend: false,
  }
]

Plotly.plot("Isolines", dt)

function Update() { // TODO update responsiveness

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