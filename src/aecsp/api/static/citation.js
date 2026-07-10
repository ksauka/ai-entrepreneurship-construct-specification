// Shared helpers for in-text citations and links out to the source article.
// Used by the dashboard, knowledge graph, and assistant pages.

function _escAttr(s) {
  return String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// Best available link for a paper: DOI first, then the Scopus record link.
function paperHref(p) {
  const doi = (p.DOI || "").trim();
  if (doi) return "https://doi.org/" + doi.replace(/^https?:\/\/(dx\.)?doi\.org\//i, "");
  const link = (p.Link || "").trim();
  return link || null;
}

// In-text citation, for example "Obschonka et al. (2019)".
function citation(p) {
  const authors = String(p.Authors || "").split(";").map(s => s.trim()).filter(Boolean);
  const year = String(p.Year || "").trim();
  const surname = (name) => (name.split(",")[0] || name).trim().split(" ")[0];
  let who;
  if (authors.length === 0) who = "Anon.";
  else if (authors.length === 1) who = surname(authors[0]);
  else if (authors.length === 2) who = surname(authors[0]) + " and " + surname(authors[1]);
  else who = surname(authors[0]) + " et al.";
  return year ? `${who} (${year})` : who;
}

// Paper title as a link when a DOI or Scopus link exists, plain text otherwise.
function paperTitleLink(p) {
  const href = paperHref(p);
  const title = _escAttr(p.Title || "(untitled)");
  return href ? `<a href="${_escAttr(href)}" target="_blank" rel="noopener">${title}</a>` : title;
}

// One evidence row: linked title plus citation and source.
function paperEvidenceItem(p) {
  return `<div class="ev-item">
    <div class="t">${paperTitleLink(p)}</div>
    <div class="m">${_escAttr(citation(p))} · ${_escAttr(p["Source title"])}</div>
  </div>`;
}
