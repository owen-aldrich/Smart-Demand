// Real Chromium smoke test using built-in Node modules (Node 22+).
// Run: node tests/browser_smoke.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const net = require('node:net');
const { spawn } = require('node:child_process');

const root = path.resolve(__dirname, '..');
const browsers = [
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
];
const browserPath = process.env.SMARTDEMAND_BROWSER || browsers.find(file => fs.existsSync(file));
if (!browserPath) throw new Error('Set SMARTDEMAND_BROWSER to a Chromium executable.');
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'smartdemand-browser-'));
const pause = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
let server, browser, socket;
let serverLog = '';
const fetchLocal = url => fetch(url, { signal: AbortSignal.timeout(3000) });

async function freePort() {
  const probe = net.createServer();
  await new Promise(resolve => probe.listen(0, '127.0.0.1', resolve));
  const port = probe.address().port;
  await new Promise(resolve => probe.close(resolve));
  return port;
}
async function until(check, description) {
  const deadline = Date.now() + 15000;
  while (Date.now() < deadline) {
    try {
      const result = await check();
      if (result) return result;
    } catch (_) {}
    await pause(100);
  }
  throw new Error('Timed out: ' + description + '\n' + serverLog);
}

async function run() {
  const port = await freePort();
  const origin = 'http://127.0.0.1:' + port;
  server = spawn(process.env.SMARTDEMAND_PYTHON || 'python', [
    '-B', '-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', String(port),
  ], { cwd: root, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
  server.stdout.on('data', data => { serverLog += data; });
  server.stderr.on('data', data => { serverLog += data; });
  server.on('error', error => { serverLog += error.message; });
  await until(async () => (await fetchLocal(origin + '/api/config')).ok, 'API startup');

  browser = spawn(browserPath, [
    '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
    '--disable-background-networking', '--disable-component-update', '--disable-sync',
    '--remote-debugging-port=0', '--user-data-dir=' + profile, 'about:blank',
  ], { windowsHide: true, stdio: ['ignore', 'ignore', 'pipe'] });
  browser.stderr.on('data', data => { serverLog += String(data).slice(-2000); });
  browser.on('error', error => { serverLog += error.message; });
  const activePort = path.join(profile, 'DevToolsActivePort');
  const debugPort = await until(() => {
    const value = fs.readFileSync(activePort, 'utf8').split(/\r?\n/)[0];
    return /^\d+$/.test(value) ? value : null;
  }, 'headless browser startup');
  const targets = await until(async () => {
    const response = await fetchLocal('http://127.0.0.1:' + debugPort + '/json/list');
    return response.json();
  }, 'browser debugging endpoint');
  const page = targets.find(target => target.type === 'page');
  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Browser WebSocket did not open')), 5000);
    socket.addEventListener('open', () => { clearTimeout(timer); resolve(); }, { once: true });
    socket.addEventListener('error', event => { clearTimeout(timer); reject(event.error || new Error('Browser socket error')); }, { once: true });
  });
  let sequence = 0;
  const pending = new Map();
  const javascriptErrors = [];
  socket.addEventListener('message', event => {
    const message = JSON.parse(String(event.data));
    if (message.method === 'Runtime.exceptionThrown') {
      javascriptErrors.push(message.params.exceptionDetails.text);
    }
    if (message.id && pending.has(message.id)) {
      const callback = pending.get(message.id);
      pending.delete(message.id);
      message.error ? callback.reject(new Error(message.error.message)) : callback.resolve(message.result);
    }
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++sequence;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error('Browser command timed out: ' + method));
    }, 5000);
    pending.set(id, {
      resolve: value => { clearTimeout(timer); resolve(value); },
      reject: error => { clearTimeout(timer); reject(error); },
    });
    socket.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async expression => {
    const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  const waitFor = expression => until(() => evaluate(expression), expression);
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', {
    width: 1280, height: 900, deviceScaleFactor: 1, mobile: false,
  });
  await send('Page.navigate', { url: origin });
  await waitFor("document.getElementById('predictBtn') && !document.getElementById('predictBtn').disabled");
  assert.equal(await evaluate("document.getElementById('category').value"), 'electronics');
  assert.equal(await evaluate("document.getElementById('salesAverage').textContent"), '90.00');
  assert.match(await evaluate("document.getElementById('historyLabel1').textContent"), /February/);
  assert.equal(await evaluate("document.getElementById('discount')"), null);

  await evaluate("document.getElementById('predictBtn').click()");
  await waitFor("document.getElementById('rcModelName').textContent === 'Random Forest' && !document.getElementById('predictBtn').disabled");
  const rfResult = await evaluate("document.getElementById('rcUnits').textContent");
  assert.match(rfResult, /^90 units in May$/);
  assert.equal(await evaluate("document.getElementById('distBars').children.length"), 4);

  await evaluate("document.getElementById('btnLR').click(); document.getElementById('predictBtn').click()");
  await waitFor("document.getElementById('rcModelName').textContent === 'Linear Regression' && !document.getElementById('predictBtn').disabled");
  const lrResult = await evaluate("document.getElementById('rcUnits').textContent");
  assert.match(lrResult, /^123 units in May$/);
  assert.equal(await evaluate("document.getElementById('distBars').children.length"), 0);
  assert.match(await evaluate("document.getElementById('distributionNote').textContent"), /No tree distribution/);

  await evaluate("document.getElementById('numProducts').value = 1000; document.getElementById('predictBtn').click()");
  await waitFor("!document.getElementById('formError').hidden && !document.getElementById('predictBtn').disabled");
  assert.match(await evaluate("document.getElementById('formError').textContent"), /Unique products/);

  await send('Emulation.setDeviceMetricsOverride', {
    width: 390, height: 844, deviceScaleFactor: 1, mobile: true,
  });
  await pause(150);
  const dimensions = await evaluate('({ width: document.documentElement.clientWidth, content: document.documentElement.scrollWidth, viewport: window.innerWidth })');
  if (dimensions.content > 391) {
    console.error(await evaluate("Array.from(document.querySelectorAll('body *')).filter(el => el.getBoundingClientRect().right > 391).map(el => ({tag: el.tagName, id: el.id, class: el.className, width: el.getBoundingClientRect().width, right: el.getBoundingClientRect().right})).slice(0, 20)"));
  }
  assert.ok(dimensions.content <= 391, JSON.stringify(dimensions));
  assert.deepEqual(javascriptErrors, []);
  console.log(JSON.stringify({ rf: rfResult, lr: lrResult, mobile: dimensions, javascriptErrors }));
}

run().catch(error => {
  console.error(error);
  process.exitCode = 1;
}).finally(async () => {
  if (socket) socket.close();
  if (browser) browser.kill();
  if (server) server.kill();
  // Only remove this test's freshly created, validated temporary profile.
  const tempRoot = path.resolve(os.tmpdir()) + path.sep;
  const resolved = path.resolve(profile);
  if (resolved.startsWith(tempRoot) && path.basename(resolved).startsWith('smartdemand-browser-')) {
    await pause(300);
    try { fs.rmSync(resolved, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 }); } catch (_) {}
  }
});
