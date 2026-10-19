// Collapse notebook cells tagged hide-input. Each run of adjacent
// tagged cells becomes one closed <details>, so the setup code stays
// one click away instead of filling the top of the page. Without
// JavaScript the cells show in full.
document.addEventListener("DOMContentLoaded", () => {
  const runs = [];
  document.querySelectorAll(".jp-Cell.celltag_hide-input").forEach((cell) => {
    // mkdocs-jupyter nests each cell in a wrapper with the same
    // classes; only the outer one is a sibling of the next cell.
    if (cell.parentElement.closest(".celltag_hide-input")) {
      return;
    }
    const run = runs[runs.length - 1];
    if (run && run[run.length - 1].nextElementSibling === cell) {
      run.push(cell);
    } else {
      runs.push([cell]);
    }
  });
  runs.forEach((run) => {
    const details = document.createElement("details");
    details.className = "hidden-setup";
    const summary = document.createElement("summary");
    summary.textContent =
      "Setup code: imports, random seeds, and plot helpers";
    details.append(summary);
    run[0].before(details);
    details.append(...run);
  });
});
