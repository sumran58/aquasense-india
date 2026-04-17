const API = "/api";

const RISK_COLOR = { CRITICAL:"#E24B4A", WARNING:"#EF9F27", STABLE:"#1D9E75", RECOVERING:"#378ADD" };
const RISK_LABEL = { CRITICAL:"Critical", WARNING:"Warning", STABLE:"Stable", RECOVERING:"Recovering" };

async function apiFetch(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(API + path, {
    method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json();
}

function toggleSidebar() { document.getElementById("sidebar").classList.toggle("open"); }
function pill(risk) { return `<span class="pill pill-${risk}">${RISK_LABEL[risk]||risk}</span>`; }
function fmt(v, d=1) { return v!=null ? Number(v).toFixed(d) : "—"; }
function updateLastUpdated(id="lastUpd") {
  const el=document.getElementById(id); if(el) el.textContent="Updated "+new Date().toLocaleTimeString();
}

async function injectDataBanner() {
  let hasReal = false;
  try {
    const d = await apiFetch('/ingest/status');
    hasReal = (d.readings_in_db||0) > 0;
  } catch {}
  const main = document.querySelector('main')||document.body;
  const ex = document.getElementById('_data_banner'); if(ex) ex.remove();
  const b = document.createElement('div'); b.id='_data_banner';
  if (hasReal) {
    b.style.cssText='background:#E1F5EE;border-bottom:1px solid rgba(29,158,117,0.25);padding:6px 28px;font-size:12px;color:#085041;display:flex;align-items:center;gap:8px;';
    b.innerHTML='<span style="width:7px;height:7px;border-radius:50%;background:#1D9E75;display:inline-block;"></span>Showing <strong>real uploaded data</strong> from your CSV.<a href="upload.html" style="margin-left:auto;color:#1D9E75;font-weight:500;text-decoration:none;">Manage uploads →</a>';
  } else {
    b.style.cssText='background:#FFF8ED;border-bottom:1px solid rgba(239,159,39,0.3);padding:6px 28px;font-size:12px;color:#633806;display:flex;align-items:center;gap:8px;';
    b.innerHTML='<span style="width:7px;height:7px;border-radius:50%;background:#EF9F27;display:inline-block;"></span>Showing <strong>demo data</strong> — no CSV uploaded yet.<a href="upload.html" style="margin-left:auto;color:#EF9F27;font-weight:500;text-decoration:none;">Upload real data →</a>';
  }
  main.insertBefore(b, main.firstChild);
}

// SAFE: runs after DOM is ready
document.addEventListener("DOMContentLoaded", function() {
  document.querySelectorAll(".nav-item").forEach(a => {
    if (a.href === location.href) a.classList.add("active");
  });
});
