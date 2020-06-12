var n = 16;
var X = [...Array(n).keys()];
var W = Array(n);
var Wvis = Array(n).fill(true);
var RandomWaveData;

var Colors = Plotly.d3.scale.linear().domain([0, n-1]).range(["blue", "red"]);

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
 
Plotly.plot("RandomWave", RandomWaveData, RandomWaveLayout);




// Isolines
var LinesData = []
for (var t = 0; t < n; t++) {
  var vx = (2 * Math.PI * n ** 2) / (n ** 2 + 4 * Math.PI ** 2 * t ** 2)
  var vy = (4 * Math.PI ** 2 * n * t) / (n ** 2 + 4 * Math.PI ** 2 * t ** 2)
  var Xl = []
  var Yl = []
  Xl.push(0)
  Xl.push(vx)
  Xl.push(NaN)
  Yl.push(0)
  Yl.push(vy)
  Yl.push(NaN)
  if (t == 0) {
    Xl.push(0)
    Xl.push(0)
    Xl.push(NaN)
    Yl.push(0)
    Yl.push(n)
    Yl.push(NaN)
    Xl.push(2 * Math.PI)
    Xl.push(2 * Math.PI)
    Xl.push(NaN)
    Yl.push(0)
    Yl.push(n)
    Yl.push(NaN)
  } else {
    for (var k = 0; k <= t; k++) {
      Xl.push(0)
      Xl.push(2 * Math.PI)
      Xl.push(NaN)
      Yl.push(k * n / t)
      Yl.push((k - 1) * n / t)
      Yl.push(NaN)
    }
  }
  LinesData.push(
    {
      name: "",
      x: Xl,
      y: Yl,
      // yaxis: 'y',
      type: 'scatter',
      mode: "lines",
      showlegend: false,
      line: {
        width: Math.max(Math.abs(W[t]) * 10, 1),
        dash: W[t] >= 0 ? "solid" : "dash",
        color: Colors(t)
      },
      // hovertemplate: "t = %{x}<br>W[%{x}] = %{y:.2f}"
    }
  )
}

var x_vals = []
var x_text = []
for (let i = -20; i < 100; i++) {
  x_vals.push(i * Math.PI)
  x_text.push("$ " + String(i) + " \\pi $")
}


var IsolinesLayout = {
  title:'Isolines',
  font: {
    family: "Computer Modern",
    size: 18,
    color: "black"
  },
  xaxis: {
    tickvals: x_vals,
    ticktext: x_text,
  },
  yaxis: {
    scaleanchor:"x",
    range:[0, n+1],
    tickvals: X,
  }
};
 
Plotly.plot("Isolines", LinesData, IsolinesLayout);

// 3D

var res_f = 200
var res_p = 200

var F = [...Array(res_f).keys()].map(i => i * n / (res_f - 1))
var P = [...Array(res_p).keys()].map(i => i * (2 * Math.PI) / (res_p - 1))
var FP

function UpdateSurface() {
  FP = Array(res_f).fill(0).map(x => Array(res_p).fill(0))
  for (var t = 0; t < n; t++) {
    if (Wvis[t]) {
      for (var i = 0; i < res_f; i++) {
        for (var j = 0; j < res_p; j++) {
          FP[i][j] += W[t] * Math.cos(P[j] + 2 * Math.PI * F[i] * t / n)
        }
      }
    }
  }
/////// Use absolute values to surface ///////
//////////////////////////////////////////////
  // for (var i = 0; i < res_f; i++) {
  //   for (var j = 0; j < res_p; j++) {
  //     FP[i][j] = Math.abs(FP[i][j])
  //   }
  // }
//////////////////////////////////////////////
//////////////////////////////////////////////
}
UpdateSurface()


var SurfaceData = [{
  x: P,
  y: F,
  z: FP,
  type: 'surface',
  showscale: false
}];

var SurfaceLayout = {
  scene:{
    xaxis:{title:"Phase"},
    yaxis:{title:"Frequency"},
    zaxis:{title:"Amplitude"},
    camera:{
      eye:{x:0, y:0, z:2}, 
      up:{x:0, y:1, z:0}
    }
  } 
}

// NewSurfaceData ={z:{FP}}


Plotly.plot("Surface", SurfaceData, SurfaceLayout);

function Update() {

  RandomWave();

  var IsolinesUpdate = {line:X.map(i => {return {width: Math.max(Math.abs(W[i]) * 10, 1), dash:W[i] >= 0 ? "solid" : "dash", color: Colors(i)}})}

  UpdateSurface()

  var NewSurfaceData = [{
    x: P,
    y: F,
    z: FP,
    type: 'surface',
    showscale: false
  }];

  Plotly.react("RandomWave", RandomWaveData, RandomWaveLayout )
  Plotly.restyle("Isolines", IsolinesUpdate)
  Plotly.react("Surface", NewSurfaceData, SurfaceLayout)
}


var RandomWaveDiv = document.getElementById("RandomWave");
// var IsolinesDiv = document.getElementById("Isolines");

RandomWaveDiv.on('plotly_doubleclick', function(data){
  var t = data.points[0].curveNumber;  
  Wvis.fill(false);
  Wvis[t] = true;
  var update = {'marker':{color: "gray"}};
  Plotly.restyle("RandomWave", update, X);
  update = {'marker':{color: "black"}};
  Plotly.restyle("RandomWave", update, [t])
  
  console.log("dc", t)
});

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

  UpdateSurface()

  var NewSurfaceData = [{
    x: P,
    y: F,
    z: FP,
    type: 'surface',
    showscale: false
  }];

  Plotly.restyle("RandomWave", RandomWaveUpdate, [t])
  Plotly.restyle("Isolines", IsolinesUpdate, [t])
  Plotly.react("Surface", NewSurfaceData, SurfaceLayout)
});

document.getElementById("UpdateBtn").onclick = function() {Update()};

document.getElementById("ShowBtn").onclick = function() {
  Wvis.fill(true)
  var RandomWaveUpdate = {'marker':{color: "black"}};
  var IsolinesUpdate = {"visible": true}

  UpdateSurface()

  var NewSurfaceData = [{
    x: P,
    y: F,
    z: FP,
    type: 'surface',
    showscale: false
  }];

  Plotly.restyle("RandomWave", RandomWaveUpdate)
  Plotly.restyle("Isolines", IsolinesUpdate)
  Plotly.react("Surface", NewSurfaceData, SurfaceLayout)
};

document.getElementById("HideBtn").onclick = function() {
  Wvis.fill(false)
  var RandomWaveUpdate = {'marker':{color: "gray"}};
  var IsolinesUpdate = {"visible": false}

  UpdateSurface()

  var NewSurfaceData = [{
    x: P,
    y: F,
    z: FP,
    type: 'surface',
    showscale: false
  }];

  Plotly.restyle("RandomWave", RandomWaveUpdate)
  Plotly.restyle("Isolines", IsolinesUpdate)
  Plotly.react("Surface", NewSurfaceData, SurfaceLayout)
};