// Shared helpers for in-text citations and links out to the source article.
// Used by the dashboard, observed composition, topic review, knowledge graph,
// and assistant pages.

function _escAttr(s) {
  return String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// Best available link for a paper: Scopus first, then DOI.
// Scopus is primary because the corpus and its citation metrics originate there.
function paperHref(p) {
  const link = (p.Link || "").trim();
  if (link) {
    try {
      const url = new URL(link);
      if (url.protocol === "https:" || url.protocol === "http:") return url.href;
    } catch (_) {
      // Invalid or relative values are not exposed as publication links.
    }
  }
  const doi = (p.DOI || "").trim();
  return doi
    ? "https://doi.org/" + doi.replace(/^https?:\/\/(dx\.)?doi\.org\//i, "")
    : null;
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
  return href ? `<a href="${_escAttr(href)}" target="_blank" rel="noopener noreferrer">${title}</a>` : title;
}

// One evidence row: linked title plus citation and source.
function paperEvidenceItem(p) {
  return `<div class="ev-item">
    <div class="t">${paperTitleLink(p)}</div>
    <div class="m">${_escAttr(citation(p))} · ${_escAttr(p["Source title"])}</div>
  </div>`;
}
