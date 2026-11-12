// Entry point: registers the lazily loaded pages, sets the footer mode text and starts the router.
import { $, app, isStatic, lazy, route, start } from './core.js';

const PAGES = { 'home.js': ['home'], 'view.js': ['view'], 'cloud.js': ['cloud'], 'filter.js': ['filter'],
  'live.js': ['live'], 'about.js': ['about'] };
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));

route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: depth maths runs in your browser (JS port of the Python code); live Kinect needs the local app'
  : 'local app: Python server on this machine';
start();
