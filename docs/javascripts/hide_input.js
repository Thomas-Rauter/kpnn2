// Collapse notebook cells tagged hide-input. Each run of adjacent
// tagged cells becomes one closed <details>, so the setup code stays
// one click away instead of filling the top of the page. Cells tagged
// hide-code collapse only their code under a label taken from their
// first-line comment; the output, such as a figure, stays visible.
// Without JavaScript the cells show in full.
function collapsed(label) {
  const details = document.createElement("details");
  details.className = "hidden-setup";
  const summary = document.createElement("summary");
  summary.textContent = label;
  details.append(summary);
  return details;
}

// mkdocs-jupyter nests each cell in a wrapper with the same
// classes; only the outer one is a sibling of the next cell.
function outerCells(tag) {
  return [...document.querySelectorAll(`.jp-Cell.celltag_${tag}`)].filter(
    (cell) => !cell.parentElement.closest(`.celltag_${tag}`),
  );
}

document.addEventListener("DOMContentLoaded", () => {
  const runs = [];
  outerCells("hide-input").forEach((cell) => {
    const run = runs[runs.length - 1];
    if (run && run[run.length - 1].nextElementSibling === cell) {
      run.push(cell);
    } else {
      runs.push([cell]);
    }
  });
  runs.forEach((run) => {
    const details = collapsed(
      "Setup code: imports, random seeds, and plot helpers",
    );
    run[0].before(details);
    details.append(...run);
  });

  // The label is the cell's first line when it is a "# ..." comment.
  outerCells("hide-code").forEach((cell) => {
    const input = cell.querySelector(".jp-Cell-inputWrapper");
    if (!input) {
      return;
    }
    const code = input.querySelector(".highlight-ipynb pre");
    const firstLine = code ? code.textContent.split("\n")[0] : "";
    const label = firstLine.startsWith("# ")
      ? firstLine.slice(2).trim()
      : "Code";
    const details = collapsed(label);
    input.before(details);
    details.append(input);
  });
});
