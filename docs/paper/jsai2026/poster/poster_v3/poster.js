/* ============ JSAI 2026 Poster — scaling, chart, QR ============ */

/* ---- fit A0 canvas to viewport (letterbox) ---- */
function fitPoster(){
  var p = document.getElementById('poster');
  if(!p) return;
  var s = Math.min(window.innerWidth / 2000, window.innerHeight / 2828);
  p.style.transform = 'scale(' + s + ')';
}
window.addEventListener('resize', fitPoster);

/* ---- Fig 4 : layer-wise F-value box plot (BERT, SST-2) ---- */
function buildChart(){
  var data = [
    {q1:-0.55,med:-0.505,q3:-0.47,lo:-0.62,hi:-0.41},
    {q1:-0.54,med:-0.50, q3:-0.46,lo:-0.61,hi:-0.40},
    {q1:-0.53,med:-0.49, q3:-0.45,lo:-0.60,hi:-0.39},
    {q1:-0.55,med:-0.505,q3:-0.46,lo:-0.61,hi:-0.40},
    {q1:-0.49,med:-0.45, q3:-0.42,lo:-0.56,hi:-0.37},
    {q1:-0.48,med:-0.44, q3:-0.41,lo:-0.55,hi:-0.36},
    {q1:-0.49,med:-0.45, q3:-0.42,lo:-0.55,hi:-0.37},
    {q1:-0.49,med:-0.45, q3:-0.42,lo:-0.55,hi:-0.37},
    {q1:-0.48,med:-0.44, q3:-0.41,lo:-0.54,hi:-0.36},
    {q1:-0.51,med:-0.47, q3:-0.44,lo:-0.57,hi:-0.39},
    {q1:-0.47,med:-0.435,q3:-0.40,lo:-0.53,hi:-0.35},
    {q1:-0.49,med:-0.45, q3:-0.42,lo:-0.55,hi:-0.37}
  ];
  var W=820,H=470,m={t:24,r:18,b:62,l:78};
  var iw=W-m.l-m.r, ih=H-m.t-m.b;
  var vmax=-0.25,vmin=-0.66;
  function y(v){return m.t + (vmax-v)/(vmax-vmin)*ih;}
  var n=data.length, slot=iw/n, bw=slot*0.52;
  var ns='http://www.w3.org/2000/svg';
  var svg=document.createElementNS(ns,'svg');
  svg.setAttribute('viewBox','0 0 '+W+' '+H);
  svg.setAttribute('class','fig');
  function el(t,a){var e=document.createElementNS(ns,t);for(var k in a)e.setAttribute(k,a[k]);return e;}

  // gridlines + y labels
  var ticks=[-0.30,-0.40,-0.50,-0.60];
  ticks.forEach(function(t){
    svg.appendChild(el('line',{x1:m.l,y1:y(t),x2:W-m.r,y2:y(t),stroke:'#ece7da','stroke-width':1.5}));
    var lb=el('text',{x:m.l-12,y:y(t)+6,'text-anchor':'end','font-size':18,fill:'#9aa0ab','font-family':'JetBrains Mono, monospace'});
    lb.textContent=t.toFixed(2); svg.appendChild(lb);
  });
  // random baseline
  svg.appendChild(el('line',{x1:m.l,y1:y(-0.515),x2:W-m.r,y2:y(-0.515),stroke:'#b0556a','stroke-width':2.5,'stroke-dasharray':'9 7'}));
  var bl=el('text',{x:W-m.r,y:y(-0.515)-10,'text-anchor':'end','font-size':17,fill:'#b0556a','font-weight':700});
  bl.textContent='Random baseline'; svg.appendChild(bl);

  // warm -> cool ramp across layers
  function lerp(a,b,t){return a+(b-a)*t;}
  function col(i){
    var t=i/(n-1);
    // explore(221,122,53) -> structure(47,99,207)
    var r=Math.round(lerp(221,47,t)), g=Math.round(lerp(122,99,t)), b=Math.round(lerp(53,207,t));
    return 'rgb('+r+','+g+','+b+')';
  }
  data.forEach(function(d,i){
    var cx=m.l+slot*(i+0.5), c=col(i);
    // whiskers
    svg.appendChild(el('line',{x1:cx,y1:y(d.hi),x2:cx,y2:y(d.lo),stroke:c,'stroke-width':2,opacity:.55}));
    svg.appendChild(el('line',{x1:cx-bw*0.3,y1:y(d.hi),x2:cx+bw*0.3,y2:y(d.hi),stroke:c,'stroke-width':2,opacity:.55}));
    svg.appendChild(el('line',{x1:cx-bw*0.3,y1:y(d.lo),x2:cx+bw*0.3,y2:y(d.lo),stroke:c,'stroke-width':2,opacity:.55}));
    // box
    svg.appendChild(el('rect',{x:cx-bw/2,y:y(d.q3),width:bw,height:y(d.q1)-y(d.q3),rx:4,fill:c,opacity:.32,stroke:c,'stroke-width':2}));
    // median
    svg.appendChild(el('line',{x1:cx-bw/2,y1:y(d.med),x2:cx+bw/2,y2:y(d.med),stroke:c,'stroke-width':3.5}));
    // x label
    var xl=el('text',{x:cx,y:H-m.b+30,'text-anchor':'middle','font-size':18,fill:'#6b7686','font-family':'JetBrains Mono, monospace'});
    xl.textContent=i; svg.appendChild(xl);
  });
  // axis lines
  svg.appendChild(el('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,stroke:'#cdc6b6','stroke-width':2}));
  svg.appendChild(el('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,stroke:'#cdc6b6','stroke-width':2}));
  // axis titles
  var xt=el('text',{x:m.l+iw/2,y:H-8,'text-anchor':'middle','font-size':19,fill:'#374055','font-weight':700});
  xt.textContent='層インデックス  Layer index'; svg.appendChild(xt);
  var yt=el('text',{x:18,y:m.t+ih/2,'text-anchor':'middle','font-size':19,fill:'#374055','font-weight':700,transform:'rotate(-90 18 '+(m.t+ih/2)+')'});
  yt.textContent='F 値 (生値)'; svg.appendChild(yt);
  // phase arrow
  var arrow=el('text',{x:m.l+iw*0.5,y:m.t+18,'text-anchor':'middle','font-size':19,fill:'#9aa0ab','font-style':'italic'});
  arrow.textContent='探索相  →  構造相  (0 に接近)'; svg.appendChild(arrow);

  document.getElementById('chart').appendChild(svg);
}

/* ---- QR codes ---- */
function buildQR(){
  var items=[
    {id:'qr-paper',url:'https://miyauchikazuyoshi.github.io/InsightSpike-AI/paper/jsai2026/C000993.pdf'},
    {id:'qr-landing',url:'https://miyauchikazuyoshi.github.io/InsightSpike-AI/'},
    {id:'qr-github',url:'https://github.com/miyauchikazuyoshi/InsightSpike-AI'}
  ];
  items.forEach(function(it){
    var box=document.getElementById(it.id);
    if(!box) return;
    try{
      if(typeof qrcode!=='undefined'){
        var qr=qrcode(0,'M'); qr.addData(it.url); qr.make();
        box.innerHTML=qr.createSvgTag({cellSize:4,margin:0,scalable:true});
        return;
      }
    }catch(e){}
    box.innerHTML='<span style="font-size:13px;color:#444;text-align:center">QR</span>';
  });
}

function init(){ fitPoster(); buildQR(); }
if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',init);
else init();
window.addEventListener('load', function(){ fitPoster(); if(!document.querySelector('#qr-paper svg')) buildQR(); });
