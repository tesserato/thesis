document.getElementById("myBtn").onclick = function() {Update()};


var RandomWaveDiv = document.getElementById("RandomWave");
var IsolinesDiv = document.getElementById("Isolines");

var n = 10;
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
  }
};
 
Plotly.plot(RandomWaveDiv, RandomWaveData, RandomWaveLayout);

RandomWaveDiv.on('plotly_doubleclick', function(data){
  var t = data.points[0].curveNumber;  
  Wvis.fill(false);
  Wvis[t] = true;
  var update = {'marker':{color: "gray"}};
  Plotly.restyle(RandomWaveDiv, update, X);
  update = {'marker':{color: "black"}};
  Plotly.restyle(RandomWaveDiv, update, [t])
  
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
  Plotly.restyle(RandomWaveDiv, RandomWaveUpdate, [t])
  Plotly.restyle(IsolinesDiv, IsolinesUpdate, [t])
  console.log("c", t)
});

function Update() {

  RandomWave();
  var nw =X.map(i => {return {width: Math.max(Math.abs(W[i]) * 10, 1), dash:W[i] >= 0 ? "solid" : "dash", color: Colors(i)}})
  var IsolinesUpdate = {line:nw}
  console.log(IsolinesUpdate)

  Plotly.react(RandomWaveDiv, RandomWaveData, RandomWaveLayout )
  Plotly.restyle(IsolinesDiv, IsolinesUpdate)
}

// Isolines
var LinesData = []
for (var t = 0; t < n; t++){
  var vx = (2 * Math.PI * n**2) / (n**2 + 4 * Math.PI**2 * t**2)
  var vy = (4 * Math.PI**2 * n * t) / (n**2 + 4 * Math.PI**2 * t**2)
  var Xl = []
  var Yl = []
  Xl.push(0)
  Xl.push(vx)
  Xl.push(NaN)
  Yl.push(0)
  Yl.push(vy)
  Yl.push(NaN)
  for (var k = 0; k <= t; k++){
    var y0 = k * n / t
    var y2pi = (k-1) * n / t
    Xl.push(0)
    Xl.push(2 * Math.PI)
    Xl.push(NaN)
    Yl.push(y0)
    Yl.push(y2pi)
    Yl.push(NaN)
  }
  LinesData.push(
    {
      name:"",
      x: Xl,
      y: Yl,
      // yaxis: 'y',
      type:'scatter',
      mode:"lines",
      showlegend: false,
      line:{
        width: Math.max(Math.abs(W[t]) * 10, 1),
        dash: W[t] >= 0 ? "solid" : "dash",
        color: Colors(t)
      },
      // hovertemplate: "t = %{x}<br>W[%{x}] = %{y:.2f}"
    }
  )

}

var IsolinesLayout = {
  title:'Isolines',
  font: {
    family: "Computer Modern",
    size: 18,
    color: "black"
  },
  yaxis: {
    scaleanchor:"x"
    // scaleratio=1
  }
};
 
Plotly.plot(IsolinesDiv, LinesData, IsolinesLayout);