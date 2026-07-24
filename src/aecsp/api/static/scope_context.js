(function () {
  "use strict";

  const TYPE_LABELS = Object.freeze({
    retrieval_scope: "Retrieval scope",
    business_domain: "Business domain",
    analytical_residual: "Analytical residual",
    exclusive_complement: "Exclusive complement",
    analytical_population: "Analytical population",
  });

  const TYPE_EXPLANATIONS = Object.freeze({
    retrieval_scope:
      "A retrieval scope follows a prespecified source-query population.",
    business_domain:
      "A business domain is assigned from the official Scopus ASJC codes of the paper's source journal. Assignment is source-level and multi-label.",
    analytical_residual:
      "An analytical residual contains papers retained in the corpus but outside the selected business-domain rows.",
    exclusive_complement:
      "An exclusive complement is the remainder after one named analytical population is removed.",
    analytical_population:
      "An analytical population combines existing corpus papers for a stated comparison; it does not add papers.",
  });

  const esc = value => String(value ?? "").replace(/[&<>"]/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;",
  }[character]));
  let domainMetadataPromise = null;

  function render(target, scope, options = {}) {
    const element = typeof target === "string"
      ? document.getElementById(target)
      : target;
    if (!element || !scope) return;

    const prefix = options.prefix || "Current dataset";
    const count = Number(scope.papers || 0).toLocaleString();
    const type = TYPE_LABELS[scope.scope_type] || "Dataset scope";
    element.textContent = `${prefix}: ${scope.label} · ${count} papers · ${type}`;
    element.title = scope.definition || "";
    element.setAttribute("aria-label", element.textContent);
  }

  function closeDialog(dialog) {
    if (typeof dialog.close === "function" && dialog.open) dialog.close();
    else dialog.removeAttribute("open");
  }

  function domainDefinitionCards(metadata) {
    return (metadata?.domains || []).map(domain => {
      const codes = Object.entries(domain.asjc_codes || {});
      const codeLine = codes.length
        ? `<p><strong>Official Scopus ASJC codes:</strong> ${codes.map(([code, label]) => `${esc(code)} (${esc(label)})`).join("; ")}</p>`
        : "";
      const rationale = domain.rationale
        ? `<p><strong>Aggregation rationale:</strong> ${esc(domain.rationale)}</p>`
        : "";
      const sources = (domain.source_titles || []).map(source =>
        `<tr><td>${esc(source.title)}</td><td>${Number(source.papers || 0).toLocaleString()}</td></tr>`
      ).join("");
      return `<details class="scope-domain-definition">
        <summary>${esc(domain.label)} <span>${Number(domain.papers || 0).toLocaleString()} papers · ${Number(domain.source_title_count || 0).toLocaleString()} represented journals</span></summary>
        <div>
          <p>${esc(domain.definition || "")}</p>
          <p><strong>Assignment type:</strong> ${esc(domain.assignment_type || "")}</p>
          ${codeLine}${rationale}
          <div class="scope-info-table-wrap"><table><thead><tr><th>Represented journal</th><th>Papers</th></tr></thead><tbody>${sources}</tbody></table></div>
        </div>
      </details>`;
    }).join("");
  }

  function loadDomainMetadata() {
    if (!domainMetadataPromise) {
      domainMetadataPromise = fetch("/api/contrasting/metadata", {
        headers: {"Accept": "application/json"},
        cache: "no-store",
      }).then(response => {
        if (!response.ok) throw new Error(`Domain metadata request failed (${response.status})`);
        return response.json();
      }).catch(() => null);
    }
    return domainMetadataPromise;
  }

  async function openDialog(select, scopes) {
    document.getElementById("datasetScopeInfoDialog")?.remove();
    const selected = scopes.find(scope => scope.id === select.value) || scopes[0];
    if (!selected) return;
    const dialog = document.createElement("dialog");
    dialog.id = "datasetScopeInfoDialog";
    dialog.className = "scope-info-dialog";
    dialog.innerHTML = `
      <div class="scope-info-heading">
        <div><p class="scope-info-kicker">Current dataset scope</p><h2>${esc(selected.label)}</h2></div>
        <button class="scope-info-close" type="button" aria-label="Close dataset-scope information">Close</button>
      </div>
      <p><strong>${Number(selected.papers || 0).toLocaleString()} papers · ${esc(TYPE_LABELS[selected.scope_type] || "Dataset scope")}</strong></p>
      <p>${esc(selected.definition || "")}</p>
      <p>${esc(TYPE_EXPLANATIONS[selected.scope_type] || "")}</p>
      <section>
        <h3>How the scope types differ</h3>
        <p><strong>Leading entrepreneurship</strong> and <strong>Additional entrepreneurship</strong> are separate journal populations. <strong>Combined entrepreneurship</strong> is their exact union. These populations are not business domains.</p>
        <p><strong>Business domains</strong> are derived only from journals already represented in the retained corpus. Papers inherit every selected ASJC-based domain assigned to their source journal, so domain rows can overlap and must not be summed. No papers are retrieved or added to fill a domain.</p>
        <p>The <strong>full corpus</strong> remains the complete baseline, including papers whose official ASJC codes fall outside the selected domain rows.</p>
      </section>
      <section id="scopeDomainRegistry"><h3>Domain registry and represented journals</h3><p>Loading the domain registry…</p></section>`;
    document.body.appendChild(dialog);
    dialog.querySelector(".scope-info-close").onclick = () => closeDialog(dialog);
    dialog.addEventListener("click", event => {
      if (event.target === dialog) closeDialog(dialog);
    });
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
    const metadata = await loadDomainMetadata();
    const registry = dialog.querySelector("#scopeDomainRegistry");
    if (!registry || !dialog.isConnected) return;
    if (!metadata) {
      registry.innerHTML = "<h3>Domain registry and represented journals</h3><p>The detailed domain registry could not be loaded.</p>";
      return;
    }
    const method = metadata.domain_methodology || {};
    registry.innerHTML = `
      <h3>Domain construction and baseline</h3>
      <p><strong>Unit:</strong> ${esc(method.unit || "")}</p>
      <p>${esc(method.construction || "")}</p>
      <p>${esc(method.classification || "")}</p>
      <p>${esc(method.overlap || "")}</p>
      <p>${esc(method.unclassified || "")}</p>
      <p><strong>Baseline:</strong> ${esc(method.baseline || "")}</p>
      <h3>Registered groups and represented journals</h3>
      ${domainDefinitionCards(metadata)}`;
  }

  function attachInfo(selectTarget, scopes) {
    const select = typeof selectTarget === "string"
      ? document.getElementById(selectTarget)
      : selectTarget;
    if (!select || !Array.isArray(scopes) || !scopes.length) return;
    const existing = select.parentElement?.querySelector("[data-scope-info]");
    if (existing) {
      existing.onclick = event => {
        event.preventDefault();
        event.stopPropagation();
        openDialog(select, scopes);
      };
      return;
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = "info-button scope-filter-info";
    button.dataset.scopeInfo = "true";
    button.setAttribute("aria-label", "Explain dataset scopes and business domains");
    button.title = "Explain dataset scopes and business domains";
    button.textContent = "i";
    button.onclick = event => {
      event.preventDefault();
      event.stopPropagation();
      openDialog(select, scopes);
    };
    select.insertAdjacentElement("beforebegin", button);
  }

  window.ScopeContext = Object.freeze({render, attachInfo});
}());
