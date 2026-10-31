"""A single-page map of one computation: every segment drawn and coloured by the rule that
priced it, so a vendor dispute can be settled by looking at it."""

from __future__ import annotations

import html
import json
from typing import Any

COLOURS = {
    "SHORT_GAP_AERIAL": "#2563eb",
    "PARKED": "#64748b",
    "ROUTED": "#16a34a",
    "DEGRADED_AERIAL": "#f59e0b",
    "REJECTED_SPEED": "#dc2626",
    "REJECTED_MAX_DISTANCE": "#dc2626",
}

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trip __TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 body{margin:0;font:14px/1.4 system-ui,sans-serif;display:flex;height:100vh}
 #map{flex:1} aside{width:380px;overflow:auto;padding:12px 16px;border-left:1px solid #ddd}
 h1{font-size:18px;margin:0 0 4px} .muted{color:#666} table{width:100%;border-collapse:collapse}
 td{padding:4px 2px;border-bottom:1px solid #eee;vertical-align:top} .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
 tr:hover{background:#f6f6f6;cursor:pointer} .km{text-align:right;white-space:nowrap}
 @media (max-width:800px){body{flex-direction:column} aside{width:auto;height:45vh;border-left:0;border-top:1px solid #ddd}}
</style></head><body>
<div id="map"></div>
<aside><h1>__TITLE__ &middot; __KM__ km</h1>
<div class="muted">__STATUS__ &middot; rule __RULE__</div>
<p id="legend"></p><table id="items"></table></aside>
<script>
const D = __DATA__;
const map = L.map('map');
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'}).addTo(map);
const pts = D.track.lat.map((la, i) => [la, D.track.lon[i]]);
const layers = [];
D.items.forEach((it, idx) => {
  const line = L.polyline(pts.slice(it.from_point, it.to_point + 1), {
    color: D.colours[it.rule] || '#000', weight: it.rule === 'SHORT_GAP_AERIAL' ? 3 : 5,
    dashArray: it.rule.startsWith('REJECTED') || it.rule === 'DEGRADED_AERIAL' ? '6 6' : null
  }).bindPopup(`<b>${it.rule}</b><br>${(it.distance_m/1000).toFixed(3)} km<br>${it.note || ''}`).addTo(map);
  layers.push(line);
});
Object.entries(D.track.labels).forEach(([i, label]) => {
  L.circleMarker(pts[+i], {radius: 6, color: '#111', fillColor: '#fff', fillOpacity: 1})
    .bindTooltip(label).addTo(map);
});
map.fitBounds(L.latLngBounds(pts).pad(0.05));
const legend = document.getElementById('legend');
Object.entries(D.totals).filter(([, v]) => v > 0).forEach(([rule, metres]) => {
  legend.insertAdjacentHTML('beforeend',
    `<div><span class="sw" style="background:${D.colours[rule]}"></span>${rule}: ${(metres/1000).toFixed(2)} km</div>`);
});
const table = document.getElementById('items');
D.items.forEach((it, idx) => {
  if (it.rule === 'SHORT_GAP_AERIAL' && it.distance_m < 50) return;
  const t = new Date(it.start_ms).toISOString().slice(11, 19);
  const row = table.insertRow();
  row.innerHTML = `<td><span class="sw" style="background:${D.colours[it.rule]}"></span>${t}</td>` +
    `<td>${it.rule}<div class="muted">${it.note || ''}</div></td><td class="km">${(it.distance_m/1000).toFixed(2)}</td>`;
  row.onclick = () => { map.fitBounds(layers[idx].getBounds().pad(0.3)); layers[idx].openPopup(); };
});
</script></body></html>"""


def render(trip_ref: str, result: dict[str, Any]) -> str:
    data = {
        "track": result.get("track", {"lat": [], "lon": [], "labels": {}}),
        "items": result.get("line_items", []),
        "totals": result.get("rule_totals_m", {}),
        "colours": COLOURS,
    }
    # JSON inside <script>: escape "</" so a value can never close the tag.
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return (
        _PAGE.replace("__DATA__", payload)
        .replace("__TITLE__", html.escape(trip_ref))
        .replace("__KM__", html.escape(str(result.get("total_km", ""))))
        .replace("__STATUS__", html.escape(str(result.get("status", ""))))
        .replace("__RULE__", html.escape(str(result.get("rule_version", ""))))
    )
