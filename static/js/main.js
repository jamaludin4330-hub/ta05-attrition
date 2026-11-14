document.addEventListener('DOMContentLoaded', () => {
  // Menu mobile
  const toggle = document.getElementById('navToggle');
  const links = document.getElementById('navLinks');
  if (toggle) {
    toggle.addEventListener('click', () => {
      const open = links.classList.toggle('open');
      toggle.setAttribute('aria-expanded', open);
    });
  }

  // Animasi muncul saat di-scroll
  const items = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('show'); io.unobserve(e.target); } });
    }, { threshold: 0.12 });
    items.forEach(el => io.observe(el));
  } else { items.forEach(el => el.classList.add('show')); }

  // Hitung naik untuk angka statistik
  document.querySelectorAll('[data-count]').forEach(el => {
    const target = parseFloat(el.dataset.count), suffix = el.dataset.suffix || '';
    const start = performance.now(), dur = 1100;
    const tick = (t) => {
      const p = Math.min((t - start) / dur, 1);
      el.textContent = (target * p).toFixed(p === 1 && Number.isInteger(target) ? 0 : 1) + suffix;
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });

  // Animasi bar (probabilitas dan feature importance)
  setTimeout(() => {
    document.querySelectorAll('[data-width]').forEach(el => { el.style.width = el.dataset.width + '%'; });
  }, 150);

  // Tombol loading saat submit + scroll ke hasil
  const form = document.getElementById('predictForm');
  if (form) {
    form.addEventListener('submit', () => {
      const btn = document.getElementById('submitBtn');
      btn.disabled = true;
      btn.querySelector('.btn-label').textContent = 'Memproses...';
      btn.querySelector('.spinner').hidden = false;
    });
    const result = document.getElementById('resultCard');
    const firstError = document.querySelector('.has-error input, .has-error select');
    if (result && window.innerWidth <= 900) result.scrollIntoView({ behavior: 'smooth', block: 'start' });
    if (firstError) firstError.focus();
  }
});