// Shared evidence-first paper inspection for construct specification and
// construct contrasting. The API supplies the coding instrument metadata so
// both pages present the same documentary basis and terminology.

function inspectionText(value, fallback = "Not recorded") {
  const text = String(value ?? "").trim();
  return text || fallback;
}

function inspectionList(value) {
  return String(value ?? "")
    .split(/[;|]/)
    .map(item => item.trim())
    .filter(Boolean);
}

function keywordChips(value) {
  const items = inspectionList(value);
  if (!items.length) return `<span class="inspection-empty">None recorded</span>`;
  return `<div class="inspection-keywords">${items.map(item =>
    `<span class="pill">${_escAttr(item)}</span>`
  ).join("")}</div>`;
}

function evidenceTypeLabel(value) {
  return ({
    stated: "Directly stated",
    inferred: "Inferred from the displayed evidence",
    absent: "Not stated within the evidence boundary",
  })[String(value || "").toLowerCase()] || "Evidence status not recorded";
}

function confidenceLabel(value) {
  const number = Number(value);
  return String(value ?? "").trim() !== "" && Number.isFinite(number)
    ? `${(number * 100).toFixed(2)}% confidence`
    : "Confidence not recorded";
}

function dimensionEvidenceMarkup(dimension) {
  const evidence = String(dimension.evidence || "").trim();
  const mechanismLogic = dimension.source_column === "ai_mechanism"
    ? String(dimension.mechanism_logic || "").trim()
    : "";
  return `<section class="inspection-rationale">
    <h4>${_escAttr(dimension.label)}: ${_escAttr(inspectionText(dimension.value, "Missing value"))}</h4>
    <p><strong>What this dimension captures:</strong> ${_escAttr(dimension.question)}</p>
    <p><strong>Interpretive purpose:</strong> ${_escAttr(dimension.diagnosis || dimension.question)}</p>
    ${evidence
      ? `<blockquote><strong>Recorded basis for this code:</strong> ${_escAttr(evidence)}</blockquote>`
      : `<p class="inspection-empty">No separate evidence excerpt was stored for this field. Review the title, abstract, and author keywords below.</p>`}
    ${mechanismLogic ? `<p><strong>Observed mechanism logic:</strong> ${_escAttr(mechanismLogic)}</p>` : ""}
    <div class="tag-row">
      <span class="pill">${_escAttr(evidenceTypeLabel(dimension.evidence_type))}</span>
      <span class="pill">${_escAttr(confidenceLabel(dimension.confidence))}</span>
    </div>
  </section>`;
}

function profileTable(dimensions) {
  return `<div class="inspection-profile-wrap"><table class="inspection-profile">
    <thead><tr><th>Dimension</th><th>Assigned code</th><th>Evidence status</th><th>Confidence</th></tr></thead>
    <tbody>${dimensions.map(dimension => `<tr>
      <th>${_escAttr(dimension.label)}</th>
      <td>${_escAttr(inspectionText(dimension.value, "Missing value"))}</td>
      <td>${_escAttr(evidenceTypeLabel(dimension.evidence_type))}</td>
      <td>${_escAttr(confidenceLabel(dimension.confidence))}</td>
    </tr>`).join("")}</tbody>
  </table></div>`;
}

function paperInspectionCard(paper, selectedColumns = []) {
  const inspection = paper._inspection || {};
  const dimensions = Array.isArray(inspection.dimensions)
    ? inspection.dimensions.map(dimension => ({
        ...dimension,
        mechanism_logic: inspection.mechanism_logic || "",
      }))
    : [];
  const selected = new Set(selectedColumns || []);
  const rationaleDimensions = dimensions.filter(dimension =>
    dimension.selected
    || selected.has(dimension.column)
    || selected.has(dimension.source_column)
  );
  const source = inspectionText(paper["Source title"]);
  const year = inspectionText(paper.Year);
  const authors = inspectionText(paper.Authors);
  const citations = String(paper["Cited by"] ?? "").trim();
  const citationCount = Number(citations);
  const citationDisplay = citations && Number.isFinite(citationCount)
    ? ` | ${citationCount.toLocaleString()} Scopus citations`
    : "";
  const datasetViews = Array.isArray(inspection.dataset_views)
    ? inspection.dataset_views.filter(Boolean)
    : [];
  const contextRows = [
    ["Paper ID", paper.paper_id],
    ["Dataset membership", datasetViews.join("; ")],
    ["Topic", paper.bertopic_topic_label],
    ["Specification issues", inspection.specification_problem],
    ["Full-text review flags", inspection.needs_full_text],
    ["Named theories", inspection.theories_mentioned],
  ].filter(([, value]) => String(value ?? "").trim());
  const rationale = rationaleDimensions.length
    ? rationaleDimensions.map(dimensionEvidenceMarkup).join("")
    : `<p class="inspection-empty">This selection has no dimension-specific code filter.</p>`;

  return `<article class="paper-inspection-card">
    <header class="paper-inspection-header">
      <h3>${paperTitleLink(paper)}</h3>
      <p>${_escAttr(citation(paper))} | ${_escAttr(source)} | ${_escAttr(year)}</p>
      <p>${_escAttr(authors)}${citationDisplay}</p>
    </header>
    <h3 class="inspection-section-title">Why this paper is included</h3>
    ${rationale}
    <details class="inspection-details" open>
      <summary>Source evidence</summary>
      <div class="inspection-source">
        <h4>Title</h4><p>${_escAttr(inspectionText(paper.Title))}</p>
        <h4>Abstract</h4><p>${_escAttr(inspectionText(paper.Abstract, "No abstract recorded"))}</p>
        <h4>Author keywords</h4>${keywordChips(paper["Author Keywords"])}
        <h4>Index keywords</h4>${keywordChips(paper["Index Keywords"])}
        <p class="inspection-boundary"><strong>Evidence boundary:</strong> ${_escAttr(inspectionText(inspection.evidence_boundary))}</p>
      </div>
    </details>
    <details class="inspection-details">
      <summary>Complete construct profile</summary>
      ${profileTable(dimensions)}
    </details>
    <details class="inspection-details">
      <summary>Bibliographic and analytical metadata</summary>
      <dl class="inspection-metadata">${contextRows.map(([label, value]) =>
        `<dt>${_escAttr(label)}</dt><dd>${_escAttr(value)}</dd>`
      ).join("")}</dl>
    </details>
  </article>`;
}
