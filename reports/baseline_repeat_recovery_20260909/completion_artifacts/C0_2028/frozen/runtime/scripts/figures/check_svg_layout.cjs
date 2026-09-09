// Optional browser render QA. Requires playwright + an installed Chromium browser.
// Usage: node check_svg_layout.cjs <browser-executable> <preview-output-directory>
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require('playwright');

(async () => {
  const root = path.resolve(__dirname, '../..');
  const output = path.resolve(process.argv[3]);
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ executablePath: process.argv[2], headless: true });
  const page = await browser.newPage({ viewport: { width: 1400, height: 920 }, deviceScaleFactor: 1 });
  const reports = [];
  try {
    for (const file of fs.readdirSync(path.join(root, 'docs/figures')).filter(f => f.endsWith('.svg')).sort()) {
      await page.setContent('<html><head><meta charset="utf-8"></head><body style="margin:0">' + fs.readFileSync(path.join(root, 'docs/figures', file), 'utf8') + '</body></html>');
      await page.evaluate(() => document.fonts.ready);
      const errors = await page.evaluate(() => {
        const svg = document.querySelector('svg');
        const view = svg.viewBox.baseVal;
        const errors = [];
        for (const text of svg.querySelectorAll('text')) {
          const r = text.getBBox();
          if (r.x < 0 || r.y < 0 || r.x + r.width > view.width || r.y + r.height > view.height) errors.push('Outside canvas: ' + text.textContent);
        }
        for (const node of svg.querySelectorAll('[data-node-group]')) {
          const bounds = node.querySelector('rect').getBBox();
          for (const text of node.querySelectorAll('text')) {
            const r = text.getBBox();
            if (r.x < bounds.x + 5 || r.y < bounds.y + 3 || r.x + r.width > bounds.x + bounds.width - 5 || r.y + r.height > bounds.y + bounds.height - 3) errors.push('Outside node: ' + text.textContent);
          }
        }
        return errors;
      });
      await page.locator('svg').screenshot({ path: path.join(output, file.replace('.svg', '.png')) });
      reports.push({ file, errors, rendered: true });
    }
    // Optional Markdown table render: marked is a presentation-only dependency.
    const { marked } = await import(pathToFileURL(require.resolve('marked')).href);
    const markdown = fs.readFileSync(path.join(root, 'reports/EXPERIMENT_CATALOG.md'), 'utf8');
    await page.setViewportSize({ width: 1600, height: 1100 });
    await page.setContent('<html><head><meta charset="utf-8"><style>body{margin:24px;font:15px "Segoe UI","Microsoft YaHei",sans-serif;color:#172b4d}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #dae2ec;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf4fc}code{font-size:13px}img{max-width:100%}</style></head><body>' + marked.parse(markdown) + '</body></html>');
    const firstTable = page.locator('table').first();
    const rows = await firstTable.locator('tbody tr').count();
    if (rows !== 14) throw new Error('Expected 14 experiment groups in rendered table');
    await firstTable.screenshot({ path: path.join(output, 'experiment_results_table.png') });
    reports.push({ file: 'reports/EXPERIMENT_CATALOG.md', table_rows: rows, errors: [], rendered: true });
  } finally { await browser.close(); }
  console.log(JSON.stringify(reports, null, 2));
  if (reports.some(r => r.errors.length)) process.exitCode = 1;
})().catch(e => { console.error(e); process.exitCode = 1; });
