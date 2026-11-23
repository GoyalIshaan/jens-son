// Chapter routing. Labs mount on entry and are destroyed on exit, so only a few WebGL contexts ever exist.
(function () {
  const chapters = [...document.querySelectorAll('article.chapter')];
  const links = [...document.querySelectorAll('.floorplan a')];
  const sel = document.getElementById('chapter-select');
  let active = [], heroStage = null;

  function mountHero(sec) {
    const host = sec.querySelector('.hero-wrap .stage'); if (!host) return;
    heroStage = new Stage(host, { theta: 0.62, phi: 0.98, radius: 15.5, target: [0, .4, 0], hint: false, spin: 0.09 });
    systolicScene(null, { R: 16, C: 16, K: 28, flow: 'os', speed: 7 }, heroStage, true);
  }
  function show(id, scroll) {
    const sec = chapters.find(c => c.id === id) || chapters[0];
    active.forEach(l => l.destroy()); active = [];
    if (heroStage) { heroStage.destroy(); heroStage = null; }
    chapters.forEach(c => { c.hidden = c !== sec; });
    links.forEach(a => { if (a.getAttribute('href') === '#' + sec.id) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
    if (sel) sel.value = sec.id;
    document.title = 'Keep the Array Busy';
    mountHero(sec);
    if (window.THREE) sec.querySelectorAll('[data-lab]').forEach(h => { const l = mountLab(h); if (l) active.push(l); });
    if (scroll) window.scrollTo({ top: 0, behavior: 'auto' });
    try { localStorage.setItem('kab-chapter', sec.id); } catch (e) {}
  }
  function route(scroll) { let id = location.hash.slice(1); if (!id) { try { id = localStorage.getItem('kab-chapter') || ''; } catch (e) {} } show(id || chapters[0].id, scroll); }
  document.addEventListener('click', e => {
    const a = e.target.closest('a[href^="#"]'); if (!a) return;
    const id = a.getAttribute('href').slice(1); if (!chapters.some(c => c.id === id)) return;
    e.preventDefault(); try { history.replaceState(null, '', '#' + id); } catch (err) {} show(id, true);
  });
  window.addEventListener('hashchange', () => route(true));
  if (sel) sel.addEventListener('change', () => { try { history.replaceState(null, '', '#' + sel.value); } catch (err) {} show(sel.value, true); });
  const retheme = () => route(false);
  if (window.matchMedia) matchMedia('(prefers-color-scheme: dark)').addEventListener('change', retheme);
  new MutationObserver(retheme).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  const start = () => route(false);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(start, start); else start();
})();
