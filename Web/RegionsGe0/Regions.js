var n = 20
var X = [...Array(n).keys()]
var W = Array(n).fill(0.0)
var Wvis = Array(n).fill(true)
// var Colors = Plotly.d3.scale.linear().domain([0, n-1]).range(["blue", "red"])
var number_of_random_sinusoids = 1

var F = Array(number_of_random_sinusoids)
var A = Array(number_of_random_sinusoids)
var P = Array(number_of_random_sinusoids)


/////// RANDOM WAVE ///////
///////////////////////////
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
};

var RandomWaveData
function GenRandomWave() {
  for (var i=0; i < number_of_random_sinusoids; i++) {
    A[i] = Math.random()
    F[i] = Math.random() * n / 2;
    P[i] = Math.random() * Math.PI;
    for (var j = 0; j < n; j++) {
      W[j] += A[i] * Math.cos(P[i] + 2.0 * Math.PI * F[i] * j / n);
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
GenRandomWave();

Plotly.plot("RandomWave", RandomWaveData, RandomWaveLayout);

///////// REGIONS /////////
///////////////////////////
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
  var RD = `rgba(191, 63, 63, ${Math.abs(W[x]) / 10})`
  var BL = `rgba(26, 150, 65, ${Math.abs(W[x]) / 10})`
  for (var k = 0; k <= x + 1; k++) {
    Xl.push(2 * Math.PI * k - Math.PI / 2)                   ; Yl.push(0)
    Xl.push(2 * Math.PI * k - 2 * Math.PI * x - Math.PI / 2) ; Yl.push(n)
    Xl.push(2 * Math.PI * k - 2 * Math.PI * x + Math.PI / 2) ; Yl.push(n)
    Xl.push(2 * Math.PI * k + Math.PI / 2)                   ; Yl.push(0)
  }
  RegionsData.push(
    {
      name: "",
      x: Xl,
      y: Yl,
      type: "scatter",
      mode: 'lines',
      line:{width:0.5, color:"black"},
      mode: "none",
      fill: "tozeroy",
      fillcolor: W[x] >= 0 ? RD : BL,
      showlegend: false,
    }
  )
  Xl = []
  Yl = []
  for (var k = 0; k <= x + 1; k++) {
  Xl.push(2 * Math.PI * k + Math.PI / 2)                     ; Yl.push(0)
  Xl.push(2 * Math.PI * k - 2 * Math.PI * x + Math.PI / 2)   ; Yl.push(n)
  Xl.push(2 * Math.PI * k - 2 * Math.PI * x + 1.5 * Math.PI) ; Yl.push(n)
  Xl.push(2 * Math.PI * k + 1.5 * Math.PI)                   ; Yl.push(0)
  }
  RegionsData.push(
    {
      name: "",
      x: Xl,
      y: Yl,
      type: "scatter",
      mode: 'lines', 
      line:{width:0.5, color:"black"},
      fill: "tozeroy",
      fillcolor: W[x] >= 0 ? BL : RD,
      showlegend: false,
    }
  )
}
RegionsData.push(
  {
    name: "Unique Space",
    x: [0, 0, Math.PI, Math.PI, 0],
    y: [0, n/2, n/2, 0, 0],
    type: "scatter",
    mode: "lines",
    line:{width:2, color:"black"},
    showlegend: false,
  }
)

Plotly.plot("Isolines", RegionsData, RegionsLayout);




var PointsData = [
  {
    name: "Max Frequency",
    x: P,
    y: F,
    type: "scatter",
    mode: "markers",
    marker: {width:10, color:"black"},
    showlegend: false,
  }
]  
Plotly.plot("Isolines", PointsData)



function Update() {

  GenRandomWave();

  for (let x = 0; x < n; x++) {
    var RD = `rgba(191, 63, 63, ${Math.abs(W[x]) / 10})`
    var BL = `rgba(26, 150, 65, ${Math.abs(W[x]) / 10})`
    RegionsData[2 * x].fillcolor     = W[x] >= 0 ? RD : BL
    RegionsData[2 * x + 1].fillcolor = W[x] >= 0 ? BL : RD
  }

  PointsData[0].x = P
  PointsData[0].y = F

  Plotly.react("RandomWave", RandomWaveData, RandomWaveLayout )
  Plotly.react("Isolines", PointsData, RegionsLayout)
  Plotly.react("Isolines", RegionsData, RegionsLayout)
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