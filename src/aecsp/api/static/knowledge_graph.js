"use strict";

const LABEL_COLOURS = Object.freeze({
  Publication: "#e74c3c",
  Author: "#3498db",
  Journal: "#95a5a6",
  Year: "#5d6d7e",
  SearchQuery: "#566573",
  Institution: "#8e44ad",
  Keyword: "#f39c12",
  Reference: "#7f8c8d",
  Topic: "#2ecc71",
  SpecificationProfile: "#bdc3c7",
  AIRole: "#c0392b",
  AIType: "#2980b9",
  Mechanism: "#27ae60",
  LevelOfAnalysis: "#d4ac0d",
  ProcessStage: "#9b59b6",
  ScopeCondition: "#1abc9c",
  DefinitionClarity: "#34495e",
  SpecificationProblem: "#e67e22",
});

const NODE_TYPES = Object.freeze([
  ["Publication", "Publication", true, true],
  ["Author", "Author", true, false],
  ["Journal", "Journal", true, false],
  ["Year", "Year", true, false],
  ["SearchQuery", "Search query", true, false],
  ["Institution", "Institution", false, false],
  ["Keyword", "Keyword", false, false],
  ["Reference", "Reference", false, false],
  ["Topic", "Topic", true, false],
  ["SpecificationProfile", "Specification profile", true, false],
  ["AIRole", "AI role", true, false],
  ["AIType", "AI type", true, false],
  ["Mechanism", "Mechanism", true, false],
  ["LevelOfAnalysis", "Level of analysis", true, false],
  ["ProcessStage", "Process stage", true, false],
  ["ScopeCondition", "Scope condition", true, false],
  ["DefinitionClarity", "Definition clarity", true, false],
  ["SpecificationProblem", "Specification problem", true, false],
]);

const RELATIONSHIP_TYPES = Object.freeze([
  ["WROTE", "Wrote", true],
  ["CO_AUTHORED_WITH", "Co-authored with", false],
  ["AFFILIATED_WITH", "Affiliated with", false],
  ["PUBLISHED_IN", "Published in", true],
  ["PUBLISHED_IN_YEAR", "Published in year", true],
  ["CAPTURED_BY", "Captured by", true],
  ["HAS_KEYWORD", "Has keyword", false],
  ["HAS_TOPIC", "Has topic", true],
  ["REFERENCES", "References", false],
  ["CITES", "Cites", true],
  ["HAS_SPECIFICATION", "Has specification", true],
  ["SPECIFIES_ROLE", "Specifies role", true],
  ["SPECIFIES_TYPE", "Specifies type", true],
  ["SPECIFIES_MECHANISM", "Specifies mechanism", true],
  ["SPECIFIES_LEVEL", "Specifies level", true],
  ["SPECIFIES_PROCESS", "Specifies process", true],
  ["SPECIFIES_SCOPE", "Specifies scope", true],
  ["HAS_DEFINITION_CLARITY", "Has definition clarity", true],
  ["HAS_SPECIFICATION_PROBLEM", "Has specification problem", true],
]);

const SPECIFICATION_DIMENSIONS = Object.freeze([
  ["", "No specification filter", ""],
  ["AIRole", "AI role", "ai_role_function"],
  ["AIType", "AI type", "ai_type_form"],
  ["Mechanism", "Mechanism", "ai_mechanism"],
  ["LevelOfAnalysis", "Level of analysis", "level_of_analysis"],
  ["ProcessStage", "Process stage", "entrepreneurial_process_stage"],
  ["ScopeCondition", "Scope condition", "scope_conditions"],
  ["DefinitionClarity", "Definition clarity", "definition_construct_clarity"],
  ["SpecificationProblem", "Specification problem", "specification_problem"],
]);

const EDGE_NAMES = Object.freeze(Object.fromEntries(
  RELATIONSHIP_TYPES.map(([type, label]) => [type, label])
));
const NODE_NAMES = Object.freeze(Object.fromEntries(
  NODE_TYPES.map(([type, label]) => [type, label])
));
const NODE_KEYS = Object.freeze({
  Publication: "id", Author: "name", Journal: "name", Year: "value",
  SearchQuery: "id", Institution: "name", Keyword: "term", Reference: "doi",
  Topic: "uid", SpecificationProfile: "id", AIRole: "name", AIType: "name",
  Mechanism: "name", LevelOfAnalysis: "name", ProcessStage: "name",
  ScopeCondition: "name", DefinitionClarity: "name", SpecificationProblem: "name",
});

let graphScope = "full_corpus";
let availableScopes = [];
let graphStatus = null;
let network = null;
let graphNodes = new vis.DataSet([]);
let graphEdges = new vis.DataSet([]);
let selectedNodeId = null;
let clickTimer = null;
let layoutFreezeTimer = null;
let layoutFrozen = false;
const pinnedNodes = new Set();

const el = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, character => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[character]);

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Graph request failed with HTTP ${response.status}`);
  return body;
}

function selectedNodeTypes() {
  return NODE_TYPES
    .filter(([type]) => el(`node_${type}`)?.checked)
    .map(([type]) => type);
}

function selectedRelationshipTypes() {
  return RELATIONSHIP_TYPES
    .filter(([type]) => el(`rel_${type}`)?.checked)
    .map(([type]) => type);
}

function buildFilterControls() {
  el("nodeTypeFilters").innerHTML = NODE_TYPES.map(([type, label, checked, locked]) =>
    `<label><input id="node_${type}" type="checkbox" ${checked ? "checked" : ""} ${locked ? "disabled" : ""}> ${esc(label)}</label>`
  ).join("");
  el("relationshipTypeFilters").innerHTML = RELATIONSHIP_TYPES.map(([type, label, checked]) =>
    `<label><input id="rel_${type}" type="checkbox" ${checked ? "checked" : ""}> ${esc(label)}</label>`
  ).join("");
  el("specificationDimension").innerHTML = SPECIFICATION_DIMENSIONS.map(([label, name]) =>
    `<option value="${label}">${esc(name)}</option>`
  ).join("");
}

function drawLegend() {
  el("legendItems").innerHTML = NODE_TYPES.map(([type, label]) =>
    `<div class="kg-legend-item"><span class="kg-legend-dot" style="background:${LABEL_COLOURS[type]}"></span><span>${esc(label)}</span></div>`
  ).join("");
}

async function initKnowledgeGraph() {
  buildFilterControls();
  drawLegend();
  wireControls();
  const [status, scopes] = await Promise.all([
    api("/api/graph/status"),
    api("/api/scopes"),
  ]);
  graphStatus = status;
  availableScopes = scopes;
  renderConnectionStatus();
  el("graphScope").innerHTML = scopes.map(scope =>
    `<option value="${esc(scope.id)}">${esc(scope.label)} (${Number(scope.papers).toLocaleString()})</option>`
  ).join("");
  ScopeContext.attachInfo("graphScope", scopes);
  updateCurrentDatasetContext();
  await loadSeed();
}

function updateCurrentDatasetContext() {
  ScopeContext.render(
    "currentDatasetContext",
    availableScopes.find(scope => scope.id === graphScope),
  );
}

function wireControls() {
  el("graphScope").addEventListener("change", async event => {
    graphScope = event.target.value;
    updateCurrentDatasetContext();
    await updateSpecificationValues();
    await loadSeed();
  });
  el("seedLimit").addEventListener("change", loadSeed);
  el("resetGraph").addEventListener("click", loadSeed);
  el("resetGraphCanvas").addEventListener("click", loadSeed);
  el("togglePhysics").addEventListener("click", toggleLayout);
  el("fitGraph").addEventListener("click", openFullscreenAndFit);
  el("exitGraphFullscreen").addEventListener("click", exitGraphFullscreen);
  document.addEventListener("fullscreenchange", syncFullscreenControls);
  el("applyGraphFilters").addEventListener("click", loadSeed);
  el("specificationDimension").addEventListener("change", updateSpecificationValues);
  el("graphSearchButton").addEventListener("click", searchGraph);
  el("graphSearch").addEventListener("keydown", event => {
    if (event.key === "Enter") {
      event.preventDefault();
      searchGraph();
    }
  });
  el("closeGraphDetails").addEventListener("click", closeDetails);
  el("runCypher").addEventListener("click", runCypher);
}

function renderConnectionStatus() {
  const connected = Boolean(graphStatus?.connected);
  const verified = Boolean(graphStatus?.read_only_verified);
  const banner = el("graphConnection");
  banner.className = `kg-connection ${connected && verified ? "ok" : "warn"}`;
  banner.textContent = connected
    ? `${graphStatus.message} Backend: Neo4j.`
    : `${graphStatus.message} Backend: bounded dataframe seed.`;
  const enabled = Boolean(graphStatus?.raw_cypher_enabled);
  el("cypherQuery").disabled = !enabled;
  el("cypherParameters").disabled = !enabled;
  el("runCypher").disabled = !enabled;
  el("cypherAvailability").textContent = enabled
    ? "Enabled because the application principal has a verified Neo4j reader role. Write and administration clauses are rejected."
    : "Disabled. Raw Cypher requires a verified Neo4j reader role and is never enabled for administrative or Community Edition credentials.";
}

async function updateSpecificationValues() {
  const label = el("specificationDimension").value;
  const valueSelect = el("specificationValue");
  const dimension = SPECIFICATION_DIMENSIONS.find(([item]) => item === label);
  if (!dimension || !dimension[2]) {
    valueSelect.disabled = true;
    valueSelect.innerHTML = '<option value="">No code filter</option>';
    return;
  }
  valueSelect.disabled = true;
  valueSelect.innerHTML = '<option value="">Loading codes...</option>';
  try {
    const values = await api(`/api/scope/${encodeURIComponent(graphScope)}/values?column=${encodeURIComponent(dimension[2])}`);
    valueSelect.innerHTML = '<option value="">Choose an observed code</option>' + values.map(value =>
      `<option value="${esc(value)}">${esc(value)}</option>`
    ).join("");
    valueSelect.disabled = false;
  } catch (error) {
    valueSelect.innerHTML = '<option value="">Codes unavailable</option>';
    setGraphMessage(error.message, "bad");
  }
}

function seedParameters() {
  const parameters = new URLSearchParams({
    scope: graphScope,
    limit: el("seedLimit").value,
    node_types: selectedNodeTypes().join(","),
    relationship_types: selectedRelationshipTypes().join(","),
  });
  const specificationLabel = el("specificationDimension").value;
  const specificationValue = el("specificationValue").value;
  if (specificationLabel && specificationValue) {
    parameters.set("specification_label", specificationLabel);
    parameters.set("specification_value", specificationValue);
  }
  return parameters;
}

async function loadSeed() {
  setLoading(true);
  closeDetails();
  pinnedNodes.clear();
  selectedNodeId = null;
  try {
    const data = await api(`/api/graph/seed?${seedParameters()}`);
    renderGraph(data, true);
    if (data.message) setGraphMessage(data.message, data.available ? "ok" : "warn");
  } catch (error) {
    renderGraph({ nodes: [], edges: [], counts: {}, backend: "unavailable" }, true);
    setGraphMessage(error.message, "bad");
  } finally {
    setLoading(false);
  }
}

function renderGraph(data, replace) {
  const nodes = uniqueById((data.nodes || []).map(toVisNode));
  const edges = uniqueById((data.edges || []).map(toVisEdge));
  if (replace) {
    graphNodes = new vis.DataSet(nodes);
    graphEdges = new vis.DataSet(edges);
  } else {
    upsertDataSet(graphNodes, nodes);
    upsertDataSet(graphEdges, edges);
  }

  if (!network || replace) {
    if (network) network.destroy();
    network = new vis.Network(el("network"), { nodes: graphNodes, edges: graphEdges }, networkOptions());
    wireNetworkEvents();
  }
  beginLayout(replace ? 180 : 100);
  updateStatusBar(data);
  if (replace) window.setTimeout(() => network?.fit({ animation: { duration: 300 } }), 80);
}

function toVisNode(node) {
  const type = node.nodeType;
  const colour = LABEL_COLOURS[type] || "#95a5a6";
  const degree = Math.max(1, Number(node.degree || 1));
  return {
    id: node.id,
    label: truncate(node.caption || node.id, type === "Publication" ? 58 : 32),
    title: esc(node.caption || node.id),
    value: 8 + Math.sqrt(degree) * 4,
    nodeType: type,
    degree,
    properties: node.properties || {},
    color: {
      background: colour,
      border: "#2c3e50",
      highlight: { background: colour, border: "#111827" },
      hover: { background: colour, border: "#111827" },
    },
    shape: type === "Publication" ? "dot" : (type === "SpecificationProfile" ? "diamond" : "box"),
    font: { size: type === "Publication" ? 13 : 11, color: "#1f2937", face: "IBM Plex Sans" },
    borderWidth: 1,
  };
}

function toVisEdge(edge) {
  const weight = Number(edge.properties?.weight || 1);
  return {
    id: edge.id || `${edge.from}:${edge.type}:${edge.to}`,
    from: edge.from,
    to: edge.to,
    relationshipType: edge.type,
    properties: edge.properties || {},
    label: EDGE_NAMES[edge.type] || edge.type,
    value: weight,
    arrows: { to: { enabled: true, scaleFactor: 0.45 } },
    color: { color: "#aab2bd", highlight: "#2c3e50", opacity: 0.75 },
    font: { size: 9, color: "#465763", face: "IBM Plex Mono", strokeWidth: 4, strokeColor: "#ffffff", align: "middle" },
    smooth: { type: "dynamic" },
  };
}

function networkOptions() {
  return {
    nodes: { scaling: { min: 8, max: 42 } },
    edges: { scaling: { min: 1, max: 5 } },
    physics: {
      barnesHut: { gravitationalConstant: -9000, springLength: 150, avoidOverlap: 0.55 },
      stabilization: { iterations: 180, updateInterval: 25 },
    },
    interaction: { hover: true, tooltipDelay: 200, navigationButtons: true, keyboard: true },
  };
}

function wireNetworkEvents() {
  network.on("click", parameters => {
    window.clearTimeout(clickTimer);
    if (!parameters.nodes.length) {
      clickTimer = window.setTimeout(loadSeed, 220);
      return;
    }
    const nodeId = parameters.nodes[0];
    clickTimer = window.setTimeout(() => focusNode(nodeId), 220);
  });
  network.on("doubleClick", parameters => {
    window.clearTimeout(clickTimer);
    if (parameters.nodes.length) expandNode(parameters.nodes[0]);
  });
  network.on("selectNode", parameters => showDetails(parameters.nodes[0]));
  network.on("deselectNode", closeDetails);
  network.on("stabilizationIterationsDone", () => freezeLayout(true));
}

function beginLayout(iterations) {
  window.clearTimeout(layoutFreezeTimer);
  layoutFrozen = false;
  updateLayoutControls();
  network.setOptions({ physics: { enabled: true } });
  network.stabilize(iterations);
  layoutFreezeTimer = window.setTimeout(() => freezeLayout(true), 3500);
}

function freezeLayout(automatic = false) {
  if (!network) return;
  window.clearTimeout(layoutFreezeTimer);
  network.setOptions({ physics: { enabled: false } });
  layoutFrozen = true;
  updateLayoutControls();
  if (!automatic) setGraphMessage("Layout frozen. Nodes will remain in place for inspection.", "ok");
}

function resumeLayout() {
  if (!network) return;
  window.clearTimeout(layoutFreezeTimer);
  network.setOptions({ physics: { enabled: true } });
  network.startSimulation();
  layoutFrozen = false;
  updateLayoutControls();
  setGraphMessage("Layout resumed. Select Freeze layout when the arrangement is useful.", "warn");
}

function toggleLayout() {
  if (layoutFrozen) resumeLayout();
  else freezeLayout();
}

function updateLayoutControls() {
  el("togglePhysics").textContent = layoutFrozen ? "Resume layout" : "Freeze layout";
  el("layoutState").textContent = layoutFrozen ? "Layout frozen" : "Layout arranging";
}

async function openFullscreenAndFit() {
  const container = el("graphFullscreen");
  try {
    if (document.fullscreenElement !== container) await container.requestFullscreen();
    window.setTimeout(() => {
      network?.redraw();
      network?.fit({ animation: { duration: 350 } });
    }, 120);
  } catch (error) {
    setGraphMessage(`Fullscreen could not be opened: ${error.message}`, "bad");
  }
}

async function exitGraphFullscreen() {
  if (document.fullscreenElement) await document.exitFullscreen();
}

function syncFullscreenControls() {
  const active = document.fullscreenElement === el("graphFullscreen");
  el("exitGraphFullscreen").hidden = !active;
  el("fitGraph").textContent = active ? "Fit visible graph" : "Fullscreen and fit";
  window.setTimeout(() => {
    network?.redraw();
    network?.fit({ animation: { duration: 250 } });
  }, 100);
}

async function focusNode(nodeId) {
  selectedNodeId = nodeId;
  showDetails(nodeId);
  if (!graphStatus?.connected) {
    focusVisibleNode(nodeId);
    return;
  }
  setLoading(true);
  try {
    const parameters = new URLSearchParams({
      scope: graphScope,
      node_id: nodeId,
      relationship_types: selectedRelationshipTypes().join(","),
    });
    const data = await api(`/api/graph/neighborhood?${parameters}`);
    renderGraph(data, true);
    selectedNodeId = nodeId;
    network.selectNodes([nodeId]);
    showDetails(nodeId);
    setGraphMessage("Focused on the selected node and its direct neighbors.", "ok");
  } catch (error) {
    focusVisibleNode(nodeId, `Neo4j focus failed: ${error.message}`);
  } finally {
    setLoading(false);
  }
}

function focusVisibleNode(nodeId, prefix = "") {
  const focusNodeRecord = graphNodes.get(nodeId);
  if (!focusNodeRecord) {
    setGraphMessage("The selected node is no longer visible. Reset the seed and try again.", "bad");
    return;
  }
  const incidentEdges = graphEdges.get({
    filter: edge => edge.from === nodeId || edge.to === nodeId,
  });
  const visibleNodeIds = new Set([nodeId]);
  incidentEdges.forEach(edge => {
    visibleNodeIds.add(edge.from);
    visibleNodeIds.add(edge.to);
  });
  const visibleNodes = graphNodes.get([...visibleNodeIds]).filter(Boolean);

  pinnedNodes.clear();
  if (network) network.destroy();
  graphNodes = new vis.DataSet(uniqueById(visibleNodes));
  graphEdges = new vis.DataSet(uniqueById(incidentEdges));
  network = new vis.Network(
    el("network"),
    { nodes: graphNodes, edges: graphEdges },
    networkOptions(),
  );
  wireNetworkEvents();
  beginLayout(100);
  selectedNodeId = nodeId;
  network.selectNodes([nodeId]);
  showDetails(nodeId);
  updateStatusBar({ backend: graphStatus?.backend || "csv" });
  const explanation = prefix ? `${prefix} ` : "";
  setGraphMessage(
    `${explanation}Focused on the selected node and ${visibleNodes.length - 1} directly connected visible nodes.`,
    prefix ? "warn" : "ok",
  );
}

async function expandNode(nodeId) {
  if (!graphStatus?.connected) {
    setGraphMessage("Neo4j is not connected. Expansion is unavailable in dataframe fallback mode.", "warn");
    return;
  }
  setLoading(true);
  try {
    const parameters = new URLSearchParams({
      scope: graphScope,
      node_id: nodeId,
      relationship_types: selectedRelationshipTypes().join(","),
    });
    const data = await api(`/api/graph/expand?${parameters}`);
    renderGraph(data, false);
    network.selectNodes([nodeId]);
    showDetails(nodeId);
    setGraphMessage(`Expanded one hop from ${graphNodes.get(nodeId)?.label || "the selected node"}.`, "ok");
  } catch (error) {
    setGraphMessage(error.message, "bad");
  } finally {
    setLoading(false);
  }
}

async function searchGraph() {
  const text = el("graphSearch").value.trim();
  const resultsBox = el("graphSearchResults");
  if (!text) {
    resultsBox.hidden = true;
    return;
  }
  if (!graphStatus?.connected) {
    setGraphMessage("Graph search requires Neo4j. The dataframe fallback shows only the current seed.", "warn");
    return;
  }
  try {
    const parameters = new URLSearchParams({
      scope: graphScope,
      text,
      node_types: selectedNodeTypes().join(","),
      limit: "20",
    });
    const results = await api(`/api/graph/search?${parameters}`);
    resultsBox.innerHTML = results.length ? results.map((node, index) =>
      `<button type="button" data-search-index="${index}"><strong>${esc(node.caption)}</strong><span>${esc(NODE_NAMES[node.nodeType] || node.nodeType)}, degree ${Number(node.degree).toLocaleString()}</span></button>`
    ).join("") : '<p>No matching graph nodes were found in this scope.</p>';
    resultsBox.hidden = false;
    resultsBox.querySelectorAll("button").forEach(button => {
      button.addEventListener("click", () => {
        resultsBox.hidden = true;
        focusNode(results[Number(button.dataset.searchIndex)].id);
      });
    });
  } catch (error) {
    setGraphMessage(error.message, "bad");
  }
}

function showDetails(nodeId) {
  const node = graphNodes.get(nodeId);
  if (!node) return;
  selectedNodeId = nodeId;
  const incident = graphEdges.get({ filter: edge => edge.from === nodeId || edge.to === nodeId });
  const relationshipCounts = countBy(incident, edge => EDGE_NAMES[edge.relationshipType] || edge.relationshipType);
  const neighborIds = new Set(incident.map(edge => edge.from === nodeId ? edge.to : edge.from));
  const neighbors = graphNodes.get([...neighborIds]).filter(Boolean);
  const neighborCounts = countBy(neighbors, item => NODE_NAMES[item.nodeType] || item.nodeType);
  const properties = Object.entries(node.properties || {})
    .filter(([, value]) => value !== "" && value != null)
    .map(([key, value]) => `<tr><th>${esc(humanize(key))}</th><td>${esc(formatProperty(value, key))}</td></tr>`)
    .join("");
  const connections = [
    ...Object.entries(neighborCounts).map(([key, value]) => `${key}: ${value}`),
    ...Object.entries(relationshipCounts).map(([key, value]) => `${key}: ${value}`),
  ];
  const publicationSummary = node.nodeType === "Publication"
    ? publicationNodeSummary(node)
    : `<div id="graphConnectedPreview" class="kg-connected-preview"><p>Loading connected papers...</p></div>`;
  const recordAction = node.nodeType === "Publication" ? "Open full record" : "View all papers";
  el("graphDetailsBody").innerHTML = `
    <p class="kg-eyebrow">${esc(NODE_NAMES[node.nodeType] || node.nodeType)}</p>
    <h2>${esc(fullNodeCaption(node))}</h2>
    <p class="kg-detail-summary">Visible degree: ${incident.length.toLocaleString()}. ${esc(connections.join(", ") || "No visible connections")}</p>
    ${publicationSummary}
    <div class="kg-action-row">
      <button class="btn btn-compact" id="detailFocus" type="button">Focus</button>
      <button class="btn btn-compact" id="detailExpand" type="button">Expand one hop</button>
      <button class="btn btn-outline btn-compact" id="detailPin" type="button">${pinnedNodes.has(nodeId) ? "Unpin" : "Pin"}</button>
      <button class="btn btn-outline btn-compact" id="detailHide" type="button">Hide</button>
      <button class="btn btn-outline btn-compact" id="detailRecord" type="button">${recordAction}</button>
    </div>
    <table class="kg-property-table"><tbody>${properties || '<tr><td>No properties available.</td></tr>'}</tbody></table>`;
  el("graphDetails").hidden = false;
  el("detailFocus").onclick = () => focusNode(nodeId);
  el("detailExpand").onclick = () => expandNode(nodeId);
  el("detailPin").onclick = () => togglePin(nodeId);
  el("detailHide").onclick = () => hideNode(nodeId);
  el("detailRecord").onclick = () => openNodeRecords(node);
  if (node.nodeType !== "Publication") loadConnectedPreview(node);
}

function fullNodeCaption(node) {
  return firstProperty(node.properties, ["Title", "title", "display_label", "topic_label", "name", "label", "term", "value", "id"])
    || node.label;
}

function firstProperty(properties, keys) {
  for (const key of keys) {
    const value = properties?.[key];
    if (value !== undefined && value !== null && String(value).trim()) return value;
  }
  return "";
}

function publicationNodeSummary(node) {
  const properties = node.properties || {};
  const paper = {
    Title: firstProperty(properties, ["Title", "title"]) || node.label,
    Authors: firstProperty(properties, ["Authors", "authors"]),
    Year: firstProperty(properties, ["Year", "year"]),
    DOI: firstProperty(properties, ["DOI", "doi"]),
    Link: firstProperty(properties, ["Link", "link"]),
  };
  const journal = firstProperty(properties, ["Journal", "journal", "Source title"]);
  const citations = firstProperty(properties, ["Citations", "citations"]);
  const queries = firstProperty(properties, ["queries", "query_sources"]);
  const specifications = SPECIFICATION_DIMENSIONS
    .filter(([, , column]) => column && properties[column])
    .map(([, name, column]) => `<div><strong>${esc(name)}:</strong> ${esc(properties[column])}</div>`)
    .join("");
  return `<div class="kg-publication-summary">
    <div>${paperTitleLink(paper)}</div>
    <div>${esc(citation(paper))}</div>
    ${journal || paper.Year ? `<div>${esc(journal)}${journal && paper.Year ? ", " : ""}${esc(paper.Year)}</div>` : ""}
    ${citations !== "" ? `<div><strong>Citations:</strong> ${esc(citations)}</div>` : ""}
    ${queries ? `<div><strong>Search queries:</strong> ${esc(queries)}</div>` : ""}
    ${specifications ? `<div class="kg-specification-summary">${specifications}</div>` : ""}
  </div>`;
}

async function loadConnectedPreview(node) {
  const target = el("graphConnectedPreview");
  if (!target) return;
  try {
    const label = node.nodeType;
    const value = nodeValue(node);
    const items = await api(`/api/scope/${encodeURIComponent(graphScope)}/connected?label=${encodeURIComponent(label)}&value=${encodeURIComponent(value)}`);
    if (selectedNodeId !== node.id || !el("graphConnectedPreview")) return;
    const preview = items.slice(0, 6).map(paper =>
      `<div class="kg-connected-paper">${paperTitleLink(paper)} <span>${esc(paper.Year || "")}</span></div>`
    ).join("");
    target.innerHTML = `<p><strong>${items.length.toLocaleString()} connected papers</strong></p>`
      + (preview || "<p>No connected papers found.</p>")
      + (items.length > 6 ? '<button id="graphViewAllConnected" class="btn btn-outline btn-compact" type="button">View all papers</button>' : "");
    const viewAll = el("graphViewAllConnected");
    if (viewAll) viewAll.onclick = () => showConnected(label, value);
  } catch (error) {
    if (target) target.innerHTML = `<p>Connected papers could not be loaded: ${esc(error.message)}</p>`;
  }
}

function togglePin(nodeId) {
  const position = network?.getPositions([nodeId])?.[nodeId];
  if (!position) {
    setGraphMessage("The selected node position is unavailable.", "bad");
    return;
  }
  if (pinnedNodes.has(nodeId)) {
    pinnedNodes.delete(nodeId);
    graphNodes.update({ id: nodeId, fixed: false, borderWidth: 1, shadow: false });
    setGraphMessage("Node unpinned. It can move when the layout is resumed.", "ok");
  } else {
    pinnedNodes.add(nodeId);
    graphNodes.update({
      id: nodeId,
      x: position.x,
      y: position.y,
      fixed: { x: true, y: true },
      borderWidth: 4,
      shadow: { enabled: true, color: "rgba(31, 41, 55, 0.35)", size: 10 },
    });
    setGraphMessage("Node pinned. It will remain fixed if the layout is resumed.", "ok");
  }
  network.redraw();
  showDetails(nodeId);
}

function hideNode(nodeId) {
  const edgeIds = graphEdges.get({ filter: edge => edge.from === nodeId || edge.to === nodeId }).map(edge => edge.id);
  graphEdges.remove(edgeIds);
  graphNodes.remove(nodeId);
  closeDetails();
  updateStatusBar({ backend: graphStatus?.backend || "csv" });
}

async function openNodeRecords(node) {
  const value = nodeValue(node);
  if (node.nodeType === "Publication") {
    await showPaper(value);
    return;
  }
  await showConnected(node.nodeType, value);
}

async function runCypher() {
  let parameters;
  try {
    parameters = JSON.parse(el("cypherParameters").value || "{}");
  } catch (error) {
    setGraphMessage("Cypher parameters must be valid JSON.", "bad");
    return;
  }
  setLoading(true);
  try {
    const data = await api("/api/graph/cypher", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: el("cypherQuery").value, parameters, limit: 500 }),
    });
    renderGraph(data, true);
    renderCypherRows(data.columns || [], data.rows || []);
    setGraphMessage(`Read-only Cypher returned ${data.rows.length.toLocaleString()} rows.`, "ok");
  } catch (error) {
    setGraphMessage(error.message, "bad");
  } finally {
    setLoading(false);
  }
}

function renderCypherRows(columns, rows) {
  if (!rows.length) {
    el("cypherRows").innerHTML = "<p>No rows returned.</p>";
    return;
  }
  el("cypherRows").innerHTML = `<div class="kg-table-scroll"><table><thead><tr>${columns.map(column => `<th>${esc(column)}</th>`).join("")}</tr></thead><tbody>${rows.slice(0, 100).map(row => `<tr>${columns.map(column => `<td>${esc(formatProperty(row[column], column))}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function updateStatusBar(data = {}) {
  const nodeCount = graphNodes.length;
  const edgeCount = graphEdges.length;
  const specLabel = el("specificationDimension")?.value;
  const specValue = el("specificationValue")?.value;
  const active = [
    `Scope: ${graphScope}`,
    `Node types: ${selectedNodeTypes().length}`,
    `Relationship types: ${selectedRelationshipTypes().length}`,
  ];
  if (specLabel && specValue) active.push(`Specification: ${specValue}`);
  if (selectedNodeId) active.push("Focused view");
  el("graphStatusBar").innerHTML = `
    <span><strong>${nodeCount.toLocaleString()}</strong> visible nodes</span>
    <span><strong>${edgeCount.toLocaleString()}</strong> visible relationships</span>
    <span>Backend: <strong>${esc(data.backend || graphStatus?.backend || "csv")}</strong></span>
    <span>${esc(active.join("; "))}</span>`;
}

function setGraphMessage(message, level) {
  const banner = el("graphConnection");
  banner.className = `kg-connection ${level}`;
  banner.textContent = message;
}

function setLoading(loading) {
  el("graphLoading").hidden = !loading;
}

function closeDetails() {
  el("graphDetails").hidden = true;
  selectedNodeId = null;
}

function upsertDataSet(dataset, records) {
  for (const record of records) {
    if (dataset.get(record.id)) dataset.update(record);
    else dataset.add(record);
  }
}

function uniqueById(records) {
  return [...new Map(records.map(record => [record.id, record])).values()];
}

function countBy(items, keyFunction) {
  return items.reduce((counts, item) => {
    const key = keyFunction(item);
    counts[key] = (counts[key] || 0) + 1;
    return counts;
  }, {});
}

function truncate(value, width) {
  const text = String(value || "");
  return text.length <= width ? text : `${text.slice(0, width - 3)}...`;
}

function humanize(value) {
  return String(value).replace(/_/g, " ").replace(/\b\w/g, letter => letter.toUpperCase());
}

function formatProperty(value, key) {
  if (Array.isArray(value)) return value.join(", ");
  if (value && typeof value === "object") return JSON.stringify(value);
  if (String(key).toLowerCase() === "abstract") return truncate(value, 500);
  return String(value ?? "");
}

function nodeValue(node) {
  if (node.nodeType === "Topic") {
    return String(
      node.properties?.display_label
      || node.properties?.label
      || node.properties?.automatic_label
      || ""
    );
  }
  const key = NODE_KEYS[node.nodeType];
  if (key && node.properties && node.properties[key] != null) return String(node.properties[key]);
  const separator = node.id.indexOf("::");
  return separator >= 0 ? decodeURIComponent(node.id.slice(separator + 2)) : node.id;
}

async function showPaper(paperId) {
  const paper = await api(`/api/paper/${encodeURIComponent(paperId)}`);
  const rows = Object.entries(paper)
    .filter(([key, value]) => value && !["convergent_papers", "contrasting_papers"].includes(key))
    .map(([key, value]) => `<div class="spec-row"><span class="k">${esc(humanize(key))}</span><span>${esc(formatProperty(value, key))}</span></div>`)
    .join("");
  const relatedSection = (title, items) => items?.length
    ? `<h3>${esc(title)}</h3>${items.map(paperEvidenceItem).join("")}`
    : "";
  el("panelBody").innerHTML = `<h2>${paperTitleLink(paper)}</h2>${rows}`
    + relatedSection("Nearest convergent papers", paper.convergent_papers)
    + relatedSection("Nearest contrasting papers", paper.contrasting_papers);
  el("panel").classList.add("open");
}

async function showConnected(label, value) {
  const items = await api(`/api/scope/${encodeURIComponent(graphScope)}/connected?label=${encodeURIComponent(label)}&value=${encodeURIComponent(value)}`);
  el("panelBody").innerHTML = `<h2>${esc(NODE_NAMES[label] || label)}: ${esc(value)} <span class="muted">${items.length.toLocaleString()} papers</span></h2>` +
    (items.length ? items.slice(0, 200).map(paperEvidenceItem).join("") : "<p>No connected papers found.</p>");
  el("panel").classList.add("open");
}

function closePanel() {
  el("panel").classList.remove("open");
}

initKnowledgeGraph().catch(error => {
  setLoading(false);
  setGraphMessage(error.message, "bad");
});
