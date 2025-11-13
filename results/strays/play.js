const fs = require('fs');
const {JSDOM} = require('jsdom');
const html = fs.readFileSync('results/test_virtual/seed_19/index.html', 'utf8');
const dom = new JSDOM(html, {runScripts: "dangerously", resources: "usable"});
const window = dom.window;
const document = window.document;
function wait(ms) { return new Promise(res => setTimeout(res, ms)); }
(async () => {
  try {
    await wait(200);
    const slider = document.getElementById('stepSlider');
    console.log('initial', document.getElementById('g0Value').textContent);
    slider.value = '71';
    slider.dispatchEvent(new window.Event('input'));
    await wait(200);
    console.log('after step 71', document.getElementById('g0Value').textContent);
    console.log('ged raw', document.getElementById('gedRawValue').textContent);
  } catch (err) {
    console.error(err);
  }
})();
