(() => {
  const WRITE_CONTROL_SELECTOR = [
    "#reviewerName",
    "#finalizeTopics",
    ".topic-label-input",
    ".topic-status-select",
    ".topic-notes-input",
    "[data-save-topic]",
    "#annotatorId",
    "#startAnnotation",
    "#confirmResume",
    "#saveDraft",
    "#submitPaper",
    "#reviewerId",
    "[data-field]",
    "[data-save-review]",
  ].join(",");

  const disableWriteControls = root => {
    const controls = [];
    if (root instanceof Element && root.matches(WRITE_CONTROL_SELECTOR)) {
      controls.push(root);
    }
    if (root.querySelectorAll) {
      controls.push(...root.querySelectorAll(WRITE_CONTROL_SELECTOR));
    }
    controls.forEach(control => {
      control.disabled = true;
      control.setAttribute("aria-disabled", "true");
      control.title = "This control is disabled for reviewer read-only access.";
    });
  };

  const showHeaderAccess = mode => {
    if (document.getElementById("platformAccessActions")) return;
    const header = document.querySelector(".app-header");
    if (!header) return;

    const actions = document.createElement("div");
    actions.id = "platformAccessActions";
    actions.className = "platform-access-actions";
    actions.setAttribute("aria-label", "Account access");

    const accessLabel = document.createElement("span");
    accessLabel.className = "platform-access-role";
    accessLabel.textContent = mode.read_only
      ? "Reviewer · read-only"
      : "Administrator";

    const form = document.createElement("form");
    form.className = "access-sign-out-form";
    form.method = "post";
    form.action = "/logout";

    const button = document.createElement("button");
    button.className = "access-sign-out";
    button.type = "submit";
    button.textContent = "Sign out";

    form.append(button);
    actions.append(accessLabel, form);
    header.append(actions);
  };

  const apply = async () => {
    try {
      const response = await fetch("/api/access-mode", {
        headers: {"Accept": "application/json"},
        cache: "no-store",
      });
      if (response.status === 401) {
        window.location.replace("/login");
        return;
      }
      if (!response.ok) return;
      const mode = await response.json();
      document.documentElement.dataset.dashboardAccessRole = mode.role || "unknown";
      showHeaderAccess(mode);
      if (!mode.read_only) {
        return;
      }
      document.documentElement.classList.add("reviewer-read-only");
      disableWriteControls(document);
      new MutationObserver(mutations => {
        mutations.forEach(mutation => {
          mutation.addedNodes.forEach(node => {
            if (node instanceof Element) disableWriteControls(node);
          });
        });
      }).observe(document.body, {childList: true, subtree: true});
    } catch (_error) {
      // Server-side access enforcement remains authoritative if this hint fails.
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", apply, {once: true});
  } else {
    apply();
  }

  window.addEventListener("pageshow", event => {
    if (event.persisted) apply();
  });
})();
