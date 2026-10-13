// Multi-Agent Graph Neural Network Register Allocation Dashboard Logic
// Supports 4 Pattern Detection Agents (R-GCN, GAT, GraphSAGE, GCN) + Consensus Arbiter, New Architectures (R-GAT, R-SAGE, R-GIN), Chaitin-Briggs & Random

let dataset = null;
let currentDatasetFile = "data.json";
let currentModel = "consensus";
let highlightDisagreements = true;
let showAttention = false;

// Graph physics & view transform state
let nodePositions = {};
let isDraggingNode = false;
let draggedNodeId = null;
let isPanning = false;
let panStartX = 0;
let panStartY = 0;
let panX = 0;
let panY = 0;
let zoom = 1.0;
let hoverNodeId = null;

// Palette for physical registers & decisions
const colorMap = {
    "R0": "#3b82f6",     // Blue
    "R1": "#10b981",     // Emerald green
    "R2": "#f59e0b",     // Amber
    "R3": "#ec4899",     // Pink
    "R4": "#8b5cf6",     // Purple
    "R5": "#06b6d4",     // Cyan
    "SPILL": "#ef4444"   // Red
};

const colorPalette = [
    "#3b82f6", "#10b981", "#f59e0b", "#ec4899",
    "#8b5cf6", "#06b6d4", "#14b8a6", "#f97316",
    "#a855f7", "#e11d48", "#84cc16", "#6366f1",
    "#0ea5e9", "#d946ef", "#eab308", "#22c55e"
];

function getNodeColor(asgn) {
    if (!asgn) return "#64748b";
    if (asgn === "SPILL") return "#ef4444";
    if (asgn.startsWith("R")) {
        const idx = parseInt(asgn.slice(1), 10);
        if (!isNaN(idx) && idx >= 0 && idx < colorPalette.length) {
            return colorPalette[idx];
        }
    }
    return colorMap[asgn] || "#64748b";
}


const defaultModelMeta = {
    "consensus": {
        name: "Consensus Ensemble",
        specialty: "Combines predictions from the available graph neural network agents."
    },
    "rgcn": {
        name: "Relational GCN (R-GCN)",
        specialty: "Relational message passing over interference and coalescing graphs."
    },
    "gat": {
        name: "Graph Attention Network (GAT)",
        specialty: "Uses attention to weight neighborhood information."
    },
    "sage": {
        name: "GraphSAGE (SAGE)",
        specialty: "Uses neighborhood aggregation to predict register allocation decisions."
    },
    "gcn": {
        name: "Graph Convolutional Network (GCN)",
        specialty: "Uses graph connectivity and node features to predict allocation decisions."
    },
    "cb": {
        name: "Chaitin-Briggs",
        specialty: "Graph-coloring allocator with optimistic spilling."
    },
    "chaitin_briggs": {
        name: "Chaitin-Briggs",
        specialty: "Graph-coloring allocator with optimistic spilling."
    },
    "random": {
        name: "Random Allocator",
        specialty: "Random register assignment baseline."
    }
};


function init() {
    try { setupTabSwitching(); } catch (e) { console.warn("setupTabSwitching:", e); }
    try { setupModelSelector(); } catch (e) { console.warn("setupModelSelector:", e); }
    try { setupToggles(); } catch (e) { console.warn("setupToggles:", e); }
    try { setupDatasetToggle(); } catch (e) { console.warn("setupDatasetToggle:", e); }
    try { setupCanvasInteractions(); } catch (e) { console.warn("setupCanvasInteractions:", e); }
    try { setupLiveAllocation(); } catch (e) { console.warn("setupLiveAllocation:", e); }

    const reloadBtn = document.getElementById("btn-reload");
    if (reloadBtn) {
        reloadBtn.addEventListener("click", () => loadData());
    }

    const resetViewBtn = document.getElementById("btn-reset-view");
    if (resetViewBtn) {
        resetViewBtn.addEventListener("click", () => resetView());
    }

    window.addEventListener("resize", handleResize);

    loadData();
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
} else {
    init();
}

function setupLiveAllocation() {
    const runButton = document.getElementById("btn-run-tac");
    const exampleButton = document.getElementById("btn-load-tac-example");
    const input = document.getElementById("tac-input");
    const registerSelect = document.getElementById("tac-register-count");
    const status = document.getElementById("tac-run-status");
    const summary = document.getElementById("tac-run-summary");
    if (!runButton || !input || !registerSelect || !status || !summary) return;

    const example = `a = 10\nb = 20\nc = a + b\nd = c * 2\ne = a + d\nreturn e`;
    if (exampleButton) exampleButton.addEventListener("click", () => { input.value = example; });

    runButton.addEventListener("click", async () => {
        const tac = input.value.trim();
        if (!tac) {
            status.className = "live-run-status error";
            status.textContent = "Please enter some Three-Address Code first.";
            return;
        }
        runButton.disabled = true;
        runButton.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running…';
        status.className = "live-run-status";
        status.textContent = "Parsing TAC, building the CFG and interference graph, and running the allocators…";
        summary.hidden = true;
        try {
            const response = await fetch("/api/allocate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ tac, registers: Number(registerSelect.value) })
            });
            const payload = await response.json();
            if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
            dataset = payload;
            currentDatasetFile = "live";
            currentModel = "consensus";
            syncActiveModelButton();
            renderDashboard();
            status.className = "live-run-status success";
            status.textContent = `Allocation complete for ${payload.nodes.length} variables. The graph, TAC, CFG, and metrics below now show your input program.`;
            const statRows = [
                ["Consensus GNN", payload.model_stats.consensus],
                ["R-GCN", payload.model_stats.rgcn],
                ["Chaitin–Briggs", payload.model_stats.cb],
                ["Random baseline", payload.model_stats.random]
            ];
            summary.innerHTML = `<h3>Live allocation summary</h3><div class="live-result-grid">${statRows.map(([name, m]) => `<div class="live-result-card"><span>${escapeLiveText(name)}</span><strong>${m ? m.spills : "—"} spills</strong><small>Spill cost: ${m ? m.spill_cost : "—"} · Moves eliminated: ${m ? m.moves_eliminated : "—"}</small></div>`).join("")}</div><p class="live-result-note">${escapeLiveText(Object.values(payload.model_status || {}).some(v => String(v).includes("untrained model")) ? "Note: at least one agent has no compatible trained checkpoint. Those predictions come from a newly initialized model and should not be treated as meaningful trained-model results." : "Compatible trained agent checkpoints were loaded where available.")}</p>`;
            summary.hidden = false;
            window.scrollTo({ top: 0, behavior: "smooth" });
        } catch (error) {
            status.className = "live-run-status error";
            status.textContent = `${error.message}. Make sure you started the app with “py -3.12 web_server.py” from the project folder, not with Python's basic http.server.`;
        } finally {
            runButton.disabled = false;
            runButton.innerHTML = '<i class="fa-solid fa-play"></i> Run Allocation';
        }
    });
}

function escapeLiveText(value) {
    return String(value).replace(/[&<>"']/g, ch => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[ch]));
}

function loadData() {
    // 1. Immediately apply embedded dataset if available for instant offline / file-protocol rendering
    if (currentDatasetFile === "comparison_data.json" && window.__COMPARISON_DATASET__) {
        dataset = window.__COMPARISON_DATASET__;
        syncActiveModelButton();
        renderDashboard();
    } else if (currentDatasetFile === "data.json" && window.__DEFAULT_DATASET__) {
        dataset = window.__DEFAULT_DATASET__;
        syncActiveModelButton();
        renderDashboard();
    }

    // 2. Fetch fresh dataset if served over HTTP/HTTPS
    if (window.location && window.location.protocol && window.location.protocol.startsWith("http")) {
        fetch(`${currentDatasetFile}?t=` + Date.now())
            .then(res => {
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                return res.json();
            })
            .then(freshData => {
                dataset = freshData;
                syncActiveModelButton();
                renderDashboard();
            })
            .catch(err => {
                console.warn(`HTTP fetch failed for ${currentDatasetFile}, using embedded/fallback data:`, err);
                if (!dataset) {
                    dataset = getFallbackData();
                    renderDashboard();
                }
            });
    } else if (!dataset) {
        dataset = getFallbackData();
        renderDashboard();
    }
}

function handleResize() {
    if (!dataset) return;
    resizeCanvas();
    drawGraph();
}

function renderDashboard() {
    if (!dataset) return;

    // Header stats
    const kBadge = document.getElementById("k-value");
    if (kBadge) kBadge.textContent = `K = ${dataset.num_registers || 4}`;

    // Metrics grid
    setElText("stat-num-vars", dataset.nodes.length);
    setElText("stat-interf-edges", dataset.interference_edges.length);
    setElText("stat-coal-edges", dataset.coalescing_edges.length);

    const disagreeCount = dataset.disagreement_nodes ? dataset.disagreement_nodes.length : 
        dataset.nodes.filter(n => n.is_disagreement).length;
    setElText("stat-disagreements", disagreeCount);

    // Update banner & active model metrics
    updateActiveModelView();

    // Render Panels
    renderTAC();
    renderCFG();
    renderTable();
    renderPatterns();
    renderAgreement();

    // Graph setup
    resizeCanvas();
    initForceLayout();
    resetView();
}

function setElText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

// -------------------------------------------------------------
// Dataset Switcher & Model Selector
// -------------------------------------------------------------
function setupDatasetToggle() {
    const btn = document.getElementById("btn-toggle-dataset");
    if (!btn) return;

    btn.addEventListener("click", () => {
        if (currentDatasetFile === "data.json") {
            currentDatasetFile = "comparison_data.json";
            const lbl = document.getElementById("lbl-dataset");
            if (lbl) lbl.textContent = "Switch to Multi-Agent Run";
            if (currentModel === "consensus") {
                currentModel = "rgcn";
            }
        } else {
            currentDatasetFile = "data.json";
            const lbl = document.getElementById("lbl-dataset");
            if (lbl) lbl.textContent = "Switch to Comparison Run";
            if (currentModel === "gin" || currentModel === "random") {
                currentModel = "consensus";
            }
        }
        syncActiveModelButton();
        loadData();
    });
}

function syncActiveModelButton() {
    const container = document.getElementById("model-buttons-container");
    if (!container) return;
    const availableModels = new Set(
        dataset && (dataset.models || dataset.model_keys)
            ? (dataset.models || dataset.model_keys)
            : Object.keys((dataset && dataset.model_stats) || {})
    );
    const buttons = Array.from(container.querySelectorAll(".model-btn"));
    buttons.forEach(b => {
        const model = b.getAttribute("data-model");
        const supported = model && getModelAliases(model).some(alias => availableModels.has(alias));
        b.hidden = !supported;
        if (b.getAttribute("data-model") === currentModel) {
            b.classList.add("active");
        } else {
            b.classList.remove("active");
        }
    });

    if (!buttons.some(b => !b.hidden && b.getAttribute("data-model") === currentModel)) {
        const firstAvailable = buttons.find(b => !b.hidden);
        if (firstAvailable) {
            currentModel = firstAvailable.getAttribute("data-model");
            firstAvailable.classList.add("active");
        }
    }
}

function setupModelSelector() {
    const container = document.getElementById("model-buttons-container");
    if (!container) return;

    container.addEventListener("click", (e) => {
        const btn = e.target.closest(".model-btn");
        if (!btn) return;

        const model = btn.getAttribute("data-model");
        if (!model || model === currentModel) return;

        currentModel = model;

        container.querySelectorAll(".model-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");

        updateActiveModelView();
        renderTAC();
        renderTable();
        renderPatterns();
        renderAgreement();
        drawGraph();
    });
}


function getModelAliases(model) {
    const aliases = [model];

    if (model === "cb") aliases.push("chaitin_briggs");
    if (model === "chaitin_briggs") aliases.push("cb");

    if (model === "rgat") aliases.push("gat");
    if (model === "gat") aliases.push("rgat");

    if (model === "rsage") aliases.push("sage");
    if (model === "sage") aliases.push("rsage");

    return aliases;
}

function getNodeAssignment(node, model) {
    if (!node) return "SPILL";
    const aliases = getModelAliases(model);
    if (node.assignments) {
        for (const a of aliases) {
            if (node.assignments[a] !== undefined) return node.assignments[a];
        }
    }
    if (model === "cb" || model === "chaitin_briggs") {
        return node.cb_assignment || "SPILL";
    }
    if (model === "consensus") {
        return node.gnn_assignment || (node.assignments ? node.assignments.consensus : "SPILL");
    }
    return node.gnn_assignment || "SPILL";
}

function getNodeModelValue(node, field, model) {
    if (!node || !node[field]) return undefined;
    for (const alias of getModelAliases(model)) {
        if (node[field][alias] !== undefined) return node[field][alias];
    }
    return undefined;
}

function getModelStats(model) {
    if (!dataset || !dataset.model_stats) return null;
    const aliases = getModelAliases(model);
    for (const a of aliases) {
        if (dataset.model_stats[a]) return dataset.model_stats[a];
    }
    return null;
}

function getModelMeta(model) {
    const aliases = getModelAliases(model);
    if (dataset && dataset.model_metadata) {
        for (const a of aliases) {
            if (dataset.model_metadata[a]) return dataset.model_metadata[a];
        }
    }
    for (const a of aliases) {
        if (defaultModelMeta[a]) return defaultModelMeta[a];
    }
    return { name: model.toUpperCase(), specialty: "Register Allocation Model" };
}

function updateActiveModelView() {
    if (!dataset) return;

    const meta = getModelMeta(currentModel);
    const stats = getModelStats(currentModel) || {
        spills: calculateModelSpills(currentModel),
        spill_cost: calculateModelSpillCost(currentModel),
        moves_eliminated: 0
    };

    setElText("current-model-title", meta.name || currentModel);
    setElText("current-model-desc", meta.specialty || "");

    setElText("meta-spills", stats.spills !== undefined ? stats.spills : 0);
    setElText("meta-cost", stats.spill_cost !== undefined ? stats.spill_cost.toFixed(1) : "0.0");
    setElText("meta-moves", stats.moves_eliminated !== undefined ? stats.moves_eliminated : 0);
    setElText("stat-active-spills", stats.spills !== undefined ? stats.spills : 0);

    const unanPct = dataset.unanimous_pct !== undefined ? 
        (dataset.unanimous_pct <= 1.0 ? (dataset.unanimous_pct * 100).toFixed(1) : dataset.unanimous_pct.toFixed(1)) + "%" : 
        "100%";
    setElText("meta-unanimous", unanPct);
}

function calculateModelSpills(model) {
    if (!dataset || !dataset.nodes) return 0;
    return dataset.nodes.filter(n => getNodeAssignment(n, model) === "SPILL").length;
}

function calculateModelSpillCost(model) {
    if (!dataset || !dataset.nodes) return 0;
    return dataset.nodes
        .filter(n => getNodeAssignment(n, model) === "SPILL")
        .reduce((sum, n) => sum + (n.spill_cost || 1.0), 0);
}

// -------------------------------------------------------------
// Toggles & Tab Switching
// -------------------------------------------------------------
function setupToggles() {
    const disagreeToggle = document.getElementById("toggle-disagreements");
    if (disagreeToggle) {
        highlightDisagreements = disagreeToggle.checked;
        disagreeToggle.addEventListener("change", (e) => {
            highlightDisagreements = e.target.checked;
            drawGraph();
        });
    }

    const attentionToggle = document.getElementById("toggle-attention");
    if (attentionToggle) {
        showAttention = attentionToggle.checked;
        attentionToggle.addEventListener("change", (e) => {
            showAttention = e.target.checked;
            drawGraph();
        });
    }
}

function setupTabSwitching() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            const targetTab = btn.getAttribute("data-tab");
            document.querySelectorAll(".tab-pane").forEach(pane => pane.classList.remove("active"));
            const targetEl = document.getElementById(targetTab);
            if (targetEl) targetEl.classList.add("active");
        });
    });
}

// -------------------------------------------------------------
// Panels: Table, IR, CFG, Patterns, Agreement
// -------------------------------------------------------------
function renderTAC() {
    const codeEl = document.getElementById("code-tac");
    if (codeEl && dataset.instructions) {
        codeEl.textContent = dataset.instructions.join("\n");
    }

    const allocationEl = document.getElementById("tac-allocation-summary");
    if (!allocationEl) return;
    allocationEl.innerHTML = "";

    const card = document.createElement("div");
    card.className = "pattern-card";
    const title = document.createElement("h4");
    title.textContent = `${getModelMeta(currentModel).name || currentModel} allocation`;
    const summary = document.createElement("p");
    summary.textContent = "Register decisions for this TAC program:";
    const assignments = document.createElement("div");
    assignments.className = "allocation-list";

    (dataset.nodes || []).forEach(node => {
        const item = document.createElement("span");
        item.className = "allocation-item";
        item.textContent = `${node.id} → ${getNodeAssignment(node, currentModel)}`;
        assignments.appendChild(item);
    });

    card.append(title, summary, assignments);
    allocationEl.appendChild(card);
}

function renderCFG() {
    const container = document.getElementById("cfg-blocks-container");
    if (!container || !dataset.cfg_blocks) return;
    container.innerHTML = "";

    dataset.cfg_blocks.forEach(block => {
        const card = document.createElement("div");
        card.className = "cfg-block-card";
        const succs = block.successors && block.successors.length ? block.successors.join(", ") : "Exit";
        card.innerHTML = `
            <div class="cfg-header">
                <span>Block ${block.block_id} ${block.label ? `(${block.label})` : ""}</span>
                <span>Loop: ${block.loop_depth} | Succs: [${succs}]</span>
            </div>
            <pre class="code-block" style="padding: 0.5rem; margin-top: 0.4rem; font-size: 0.8rem;"><code>${(block.instructions || []).join("\n")}</code></pre>
        `;
        container.appendChild(card);
    });
}

function renderTable() {
    const body = document.getElementById("table-node-body");
    if (!body || !dataset.nodes) return;
    body.innerHTML = "";

    dataset.nodes.forEach(n => {
        const row = document.createElement("tr");

        const asgn = getNodeAssignment(n, currentModel);
        const bg = getNodeColor(asgn);

        const spillProbability = getNodeModelValue(n, "spill_probs", currentModel);
        const confidence = getNodeModelValue(n, "confidence", currentModel);
        const spillProb = spillProbability !== undefined ?
            `${(spillProbability * 100).toFixed(0)}%` :
            (asgn === "SPILL" ? "100%" : "0%");

        const conf = confidence !== undefined ?
            `${(confidence * 100).toFixed(0)}%` : "N/A";

        const disagreeTag = n.is_disagreement ? 
            `<span title="Models disagree on this node" style="color: #f59e0b; margin-left: 4px;">⚠</span>` : "";

        row.innerHTML = `
            <td><strong>${n.id}</strong>${disagreeTag}</td>
            <td>${n.loop_depth !== undefined ? n.loop_depth : 0}</td>
            <td>${(n.spill_cost || 0).toFixed(1)}</td>
            <td>${n.degree !== undefined ? n.degree : 0}</td>
            <td><span class="badge-reg" style="background:${bg}25; color:${bg}; border:1px solid ${bg}">${asgn}</span></td>
            <td>${spillProb}</td>
            <td>${conf}</td>
        `;

        row.addEventListener("mouseenter", () => {
            hoverNodeId = n.id;
            drawGraph();
        });
        row.addEventListener("mouseleave", () => {
            hoverNodeId = null;
            drawGraph();
        });

        body.appendChild(row);
    });
}

function renderPatterns() {
    const container = document.getElementById("patterns-container");
    if (!container) return;
    container.innerHTML = "";

    const selectedCard = document.createElement("div");
    selectedCard.className = "pattern-card selected-model-card";
    const selectedTitle = document.createElement("h4");
    selectedTitle.textContent = `Selected allocator: ${getModelMeta(currentModel).name || currentModel}`;
    const selectedDescription = document.createElement("p");
    selectedDescription.textContent = getModelMeta(currentModel).specialty || "";
    const selectedStats = getModelStats(currentModel) || {
        spills: calculateModelSpills(currentModel),
        spill_cost: calculateModelSpillCost(currentModel)
    };
    const selectedSummary = document.createElement("p");
    selectedSummary.textContent =
        `Spills: ${selectedStats.spills ?? 0} · Spill cost: ${(selectedStats.spill_cost || 0).toFixed(1)}`;
    selectedCard.append(selectedTitle, selectedDescription, selectedSummary);

    const activeExplanation = getModelAliases(currentModel)
        .map(alias => dataset.explanations && dataset.explanations[alias])
        .find(Boolean);
    if (activeExplanation && activeExplanation.specialty) {
        const explanation = document.createElement("p");
        explanation.textContent = `Pattern insight: ${activeExplanation.specialty}`;
        selectedCard.appendChild(explanation);
    }

    container.appendChild(selectedCard);

    if (!dataset.explanations) {
        if (dataset.model_stats) {
            const models = [
                { key: "rgcn", title: "R-GCN (Relational GCN)", desc: "Multi-relational message passing over both interference & coalescing graphs.", color: "#38bdf8", icon: "fa-arrows-split-up-and-left" },
                { key: "rgat", title: "R-GAT (Relational GAT)", desc: "Learns attention weights across interference & move edges dynamically.", color: "#a855f7", icon: "fa-eye" },
                { key: "rsage", title: "R-SAGE (Relational GraphSAGE)", desc: "Mean neighborhood sampling & inductive aggregation over relations.", color: "#10b981", icon: "fa-network-wired" },
                { key: "gin", title: "R-GIN (Relation-Aware GIN)", desc: "Sum aggregation with relation-specific MLP update for maximum expressive power.", color: "#f59e0b", icon: "fa-shapes" },
                { key: "chaitin_briggs", title: "Chaitin-Briggs", desc: "Heuristic Kempe-chain graph coloring with optimistic spilling.", color: "#60a5fa", icon: "fa-shield-halved" },
                { key: "random", title: "Random Baseline", desc: "Uniform random assignment of registers and spills.", color: "#94a3b8", icon: "fa-dice" }
            ];
            models.forEach(m => {
                const stats = dataset.model_stats[m.key] || dataset.model_stats[m.key === "chaitin_briggs" ? "cb" : m.key];
                if (!stats) return;
                const card = document.createElement("div");
                card.className = "pattern-card";
                card.innerHTML = `
                    <h4 style="color:${m.color}"><i class="fa-solid ${m.icon}"></i> ${m.title}</h4>
                    <p style="margin-bottom:0.4rem;">${m.desc}</p>
                    <p><strong>Spills:</strong> ${stats.spills} | <strong>Spill Cost:</strong> ${(stats.spill_cost || 0).toFixed(1)}</p>
                    <p><strong>Moves Eliminated:</strong> ${stats.moves_eliminated || 0} (${(stats.move_elim_rate_pct || 0).toFixed(1)}%)</p>
                    <p style="font-size:0.75rem; color:var(--text-muted); margin-top:0.2rem;">Inference latency: ${(stats.inference_time_ms || 0).toFixed(2)} ms</p>
                `;
                container.appendChild(card);
            });
            return;
        }

        container.innerHTML = `<p style="color:var(--text-secondary); padding:1rem;">No agent pattern diagnostics available for this program.</p>`;
        return;
    }

    const agentMeta = [
        { key: "rgcn", title: "Agent 1: RelationalAgent (R-GCN)", icon: "fa-arrows-split-up-and-left", color: "#38bdf8" },
        { key: "gat", title: "Agent 2: AttentionAgent (GAT)", icon: "fa-eye", color: "#a855f7" },
        { key: "sage", title: "Agent 3: NeighbourhoodAgent (GraphSAGE)", icon: "fa-network-wired", color: "#10b981" },
        { key: "gcn", title: "Agent 4: PressureAgent (GCN)", icon: "fa-compress", color: "#f59e0b" }
    ];

    agentMeta.forEach(m => {
        const exp = dataset.explanations[m.key];
        if (!exp) return;

        const card = document.createElement("div");
        card.className = "pattern-card";

        let bodyHtml = `<p><strong>Specialty:</strong> ${exp.specialty || "Compiler pattern recognition"}</p>`;

        if (m.key === "rgcn") {
            const chains = exp.move_chains_detected || [];
            const chainsText = chains.length ? chains.map(c => c.join(" → ")).join(", ") : "None";
            bodyHtml += `
                <p style="margin-top:0.4rem;"><strong>Move Chains Detected:</strong> ${exp.move_chain_count || chains.length}</p>
                <p style="font-family:var(--font-code); font-size:0.75rem; color:#93c5fd;">${chainsText}</p>
                <p style="margin-top:0.3rem;"><strong>Relational Edge Ratio:</strong> ${((exp.relational_edge_ratio || 0) * 100).toFixed(1)}%</p>
            `;
        } else if (m.key === "gat") {
            const pressNodes = (exp.highest_pressure_nodes || []).slice(0, 4);
            const pressList = pressNodes.map(p => `${p.var_name} (pressure: ${p.inward_pressure_score.toFixed(2)})`).join(", ");
            bodyHtml += `
                <p style="margin-top:0.4rem;"><strong>Top Inward Pressure Nodes:</strong></p>
                <p style="font-family:var(--font-code); font-size:0.75rem; color:#d8b4fe;">${pressList || "None"}</p>
                <p style="margin-top:0.3rem; font-size:0.75rem; color:var(--text-muted);">Attention weights indicate which interfering neighbors forced spill decisions.</p>
            `;
        } else if (m.key === "sage") {
            const hubs = (exp.hub_nodes || []).map(h => `${h.var_name} (deg: ${h.degree})`).join(", ");
            bodyHtml += `
                <p style="margin-top:0.4rem;"><strong>Hub Nodes:</strong> <span style="font-family:var(--font-code); font-size:0.75rem; color:#6ee7b7;">${hubs || "None"}</span></p>
                <p style="margin-top:0.2rem;"><strong>Max K-Core:</strong> ${exp.max_k_core || 0} | <strong>Clustering Coeff:</strong> ${(exp.avg_clustering_coefficient || 0).toFixed(3)}</p>
                <p style="margin-top:0.2rem;"><strong>Dense Forcing Cliques:</strong> ${exp.forcing_cliques_count || 0}</p>
            `;
        } else if (m.key === "gcn") {
            const hotNodes = (exp.loop_hot_nodes || []).slice(0, 5).map(h => `${h.var_name} (loop: ${h.loop_depth})`).join(", ");
            bodyHtml += `
                <p style="margin-top:0.4rem;"><strong>Global Pressure Peak:</strong> <span style="color:#fcd34d;">${(exp.global_pressure_peak || 0).toFixed(2)}x K</span></p>
                <p style="margin-top:0.2rem;"><strong>Loop-Hot Variables:</strong> <span style="font-family:var(--font-code); font-size:0.75rem; color:#fcd34d;">${hotNodes || "None"}</span></p>
            `;
        }

        card.innerHTML = `
            <h4 style="color:${m.color}"><i class="fa-solid ${m.icon}"></i> ${m.title}</h4>
            ${bodyHtml}
        `;
        container.appendChild(card);
    });
}

function renderAgreement() {
    const container = document.getElementById("agreement-container");
    if (!container) return;
    container.innerHTML = "";

    const availableModels = dataset.models || dataset.model_keys ||
        Object.keys(dataset.model_stats || {});
    const activeAliases = getModelAliases(currentModel);
    const activeKey = availableModels.find(model => activeAliases.includes(model)) || currentModel;
    const comparisons = availableModels
        .filter(model => model !== activeKey)
        .map(model => {
            let registerMatches = 0;
            let spillMatches = 0;
            let compared = 0;
            (dataset.nodes || []).forEach(node => {
                const activeAssignment = getNodeAssignment(node, activeKey);
                const otherAssignment = getNodeAssignment(node, model);
                if (activeAssignment === undefined || otherAssignment === undefined) return;
                compared += 1;
                if (activeAssignment === otherAssignment) registerMatches += 1;
                if ((activeAssignment === "SPILL") === (otherAssignment === "SPILL")) spillMatches += 1;
            });
            return {
                model,
                registerAgreement: compared ? (registerMatches / compared) * 100 : 0,
                spillAgreement: compared ? (spillMatches / compared) * 100 : 0
            };
        });

    const activeAgreementCard = document.createElement("div");
    activeAgreementCard.className = "table-wrapper";
    const activeAgreementTitle = document.createElement("div");
    activeAgreementTitle.className = "pattern-card";
    const activeModelHeading = document.createElement("h4");
    activeModelHeading.textContent = `${getModelMeta(currentModel).name || currentModel} vs. other allocators`;
    const activeModelDescription = document.createElement("p");
    activeModelDescription.textContent = "Agreement is recalculated from the current program's register assignments.";
    activeAgreementTitle.append(activeModelHeading, activeModelDescription);
    activeAgreementCard.appendChild(activeAgreementTitle);

    const activeAgreementTable = document.createElement("table");
    activeAgreementTable.className = "agreement-table";
    activeAgreementTable.innerHTML = `
        <thead><tr><th>Compared allocator</th><th>Register agreement</th><th>Spill agreement</th></tr></thead>
        <tbody></tbody>
    `;
    const activeAgreementBody = activeAgreementTable.querySelector("tbody");
    comparisons.forEach(comparison => {
        const row = document.createElement("tr");
        const nameCell = document.createElement("td");
        nameCell.textContent = getModelMeta(comparison.model).name || comparison.model;
        const registerCell = document.createElement("td");
        registerCell.textContent = `${comparison.registerAgreement.toFixed(1)}%`;
        const spillCell = document.createElement("td");
        spillCell.textContent = `${comparison.spillAgreement.toFixed(1)}%`;
        row.append(nameCell, registerCell, spillCell);
        activeAgreementBody.appendChild(row);
    });
    activeAgreementCard.appendChild(activeAgreementTable);
    container.appendChild(activeAgreementCard);

    const unanPct = dataset.unanimous_pct !== undefined ? 
        (dataset.unanimous_pct <= 1.0 ? dataset.unanimous_pct * 100 : dataset.unanimous_pct) : null;

    if (unanPct !== null) {
        const unanCard = document.createElement("div");
        unanCard.className = "pattern-card";
        unanCard.style.borderLeft = "4px solid #10b981";
        unanCard.innerHTML = `
            <h4 style="color:#10b981;"><i class="fa-solid fa-handshake"></i> Unanimous Model Agreement: ${unanPct.toFixed(1)}%</h4>
            <p>Proportion of variables where all evaluated models independently agreed on the exact same allocation decision (Spill vs Physical Register).</p>
        `;
        container.appendChild(unanCard);
    }

    // Pairwise Table
    if (dataset.pairwise_agreement) {
        const tableCard = document.createElement("div");
        tableCard.className = "table-wrapper";

        let rowsHtml = "";
        const pairs = Object.entries(dataset.pairwise_agreement);
        pairs.forEach(([pairKey, score]) => {
            const displayPair = pairKey.replace("_vs_", " ↔ ").toUpperCase();
            const pct = score <= 1.0 ? score * 100 : score;
            const badgeColor = pct >= 90 ? "#10b981" : (pct >= 70 ? "#3b82f6" : "#f59e0b");
            rowsHtml += `
                <tr>
                    <td><strong>${displayPair}</strong></td>
                    <td>
                        <div style="display:flex; align-items:center; gap:0.6rem;">
                            <div style="flex:1; background:rgba(255,255,255,0.06); height:8px; border-radius:4px; overflow:hidden;">
                                <div style="width:${pct}%; background:${badgeColor}; height:100%;"></div>
                            </div>
                            <span style="font-weight:700; color:${badgeColor}; min-width:48px;">${pct.toFixed(1)}%</span>
                        </div>
                    </td>
                </tr>
            `;
        });

        tableCard.innerHTML = `
            <div class="pattern-card" style="margin-bottom:0.5rem;">
                <h4 style="color:#38bdf8;"><i class="fa-solid fa-scale-balanced"></i> Pairwise Allocation Agreement</h4>
                <p>Percentage of program variables assigned the exact same physical register across model pairs.</p>
            </div>
            <table class="agreement-table">
                <thead>
                    <tr>
                        <th>Model Pair</th>
                        <th>Agreement Rate</th>
                    </tr>
                </thead>
                <tbody>
                    ${rowsHtml}
                </tbody>
            </table>
        `;
        container.appendChild(tableCard);
    }

    // Coalescing Jaccard Matrix
    if (dataset.register_agreement_matrix) {
        const registerCard = document.createElement("div");
        registerCard.className = "table-wrapper";
        const models = Object.keys(dataset.register_agreement_matrix);
        const thead = "<tr><th>Model</th>" + models.map(m => `<th>${m.toUpperCase()}</th>`).join("") + "</tr>";
        const rows = models.map(m1 => {
            const cells = models.map(m2 => {
                const value = dataset.register_agreement_matrix[m1][m2];
                return `<td>${Number(value).toFixed(1)}%</td>`;
            }).join("");
            return `<tr><td><strong>${m1.toUpperCase()}</strong></td>${cells}</tr>`;
        }).join("");
        registerCard.innerHTML = `
            <div class="pattern-card" style="margin-bottom:0.5rem;">
                <h4 style="color:#38bdf8;"><i class="fa-solid fa-scale-balanced"></i> Register Agreement Matrix</h4>
                <p>Pairwise percentage of variables assigned the same physical register.</p>
            </div>
            <table class="agreement-table"><thead>${thead}</thead><tbody>${rows}</tbody></table>
        `;
        container.appendChild(registerCard);
    }

    if (dataset.coalescing_jaccard_matrix) {
        const jaccardCard = document.createElement("div");
        jaccardCard.className = "table-wrapper";
        const models = Object.keys(dataset.coalescing_jaccard_matrix);
        let thead = "<tr><th>Model</th>" + models.map(m => `<th>${m.toUpperCase()}</th>`).join("") + "</tr>";
        let rows = "";
        models.forEach(m1 => {
            let row = `<tr><td><strong>${m1.toUpperCase()}</strong></td>`;
            models.forEach(m2 => {
                const val = (dataset.coalescing_jaccard_matrix[m1][m2] * 100).toFixed(0);
                row += `<td>${val}%</td>`;
            });
            row += "</tr>";
            rows += row;
        });
        jaccardCard.innerHTML = `
            <div class="pattern-card" style="margin-bottom:0.5rem; margin-top:1rem;">
                <h4 style="color:#10b981;"><i class="fa-solid fa-scissors"></i> Coalescing Move Overlap (Jaccard Index)</h4>
                <p>Overlap of eliminated MOVE instructions between model pairs (100% = identical set of coalesced moves).</p>
            </div>
            <table class="agreement-table">
                <thead>${thead}</thead>
                <tbody>${rows}</tbody>
            </table>
        `;
        container.appendChild(jaccardCard);
    }

    // Spill Concordance Matrix
    if (dataset.spill_agreement_matrix) {
        const spillAgreeCard = document.createElement("div");
        spillAgreeCard.className = "table-wrapper";
        const models = Object.keys(dataset.spill_agreement_matrix);
        let thead = "<tr><th>Model</th>" + models.map(m => `<th>${m.toUpperCase()}</th>`).join("") + "</tr>";
        let rows = "";
        models.forEach(m1 => {
            let row = `<tr><td><strong>${m1.toUpperCase()}</strong></td>`;
            models.forEach(m2 => {
                const val = dataset.spill_agreement_matrix[m1][m2].toFixed(1);
                row += `<td>${val}%</td>`;
            });
            row += "</tr>";
            rows += row;
        });
        spillAgreeCard.innerHTML = `
            <div class="pattern-card" style="margin-bottom:0.5rem; margin-top:1rem;">
                <h4 style="color:#f59e0b;"><i class="fa-solid fa-boxes-packing"></i> Spill Decision Concordance Matrix</h4>
                <p>Pairwise agreement rate (%) specifically on whether a variable is allocated to a physical register or spilled to stack.</p>
            </div>
            <table class="agreement-table">
                <thead>${thead}</thead>
                <tbody>${rows}</tbody>
            </table>
        `;
        container.appendChild(spillAgreeCard);
    }

    // Disagreement Detail
    if (dataset.disagreement_nodes && dataset.disagreement_nodes.length > 0) {
        const disCard = document.createElement("div");
        disCard.className = "pattern-card";
        disCard.style.borderLeft = "4px solid #f59e0b";
        disCard.style.marginTop = "1rem";

        let disList = "";
        dataset.disagreement_nodes.forEach(d => {
            const votesObj = d.agent_votes || d.votes || {};
            const votes = Object.entries(votesObj)
                .map(([agent, vote]) => `<span style="color:#94a3b8;">${agent.toUpperCase()}:</span> <strong>${vote}</strong>`)
                .join(" | ");
            const consensusDec = d.consensus_decision ? `<br/><span style="color:#10b981; font-weight:600;">Consensus Decided: ${d.consensus_decision}</span>` : "";
            disList += `
                <li style="margin-bottom:0.5rem;">
                    <strong>Variable ${d.var_name}</strong> (Cost: ${d.spill_cost}, Deg: ${d.degree})<br/>
                    <span style="font-size:0.75rem;">Decisions: ${votes}</span>
                    ${consensusDec}
                </li>
            `;
        });

        disCard.innerHTML = `
            <h4 style="color:#f59e0b;"><i class="fa-solid fa-triangle-exclamation"></i> Disagreement Case Breakdown</h4>
            <ul style="list-style:none; margin:0; padding:0;">${disList}</ul>
        `;
        container.appendChild(disCard);
    }
}

// -------------------------------------------------------------
// Interactive Force Layout Graph Engine
// -------------------------------------------------------------
function resizeCanvas() {
    const canvas = document.getElementById("graph-canvas");
    if (!canvas) return;
    const container = document.getElementById("canvas-container");
    const rect = container ? container.getBoundingClientRect() : { width: 800, height: 520 };

    const width = Math.max(rect.width, 300);
    const height = Math.max(rect.height || 520, 400);

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
}

function initForceLayout() {
    if (!dataset || !dataset.nodes) return;
    const canvas = document.getElementById("graph-canvas");
    const width = parseFloat(canvas.style.width) || 800;
    const height = parseFloat(canvas.style.height) || 520;

    const N = dataset.nodes.length;
    if (!N) {
        nodePositions = {};
        return;
    }
    const nodeIndices = new Map(dataset.nodes.map((node, index) => [node.id, index]));
    const radius = Math.min(width, height) * 0.32;
    const centerX = width / 2;
    const centerY = height / 2;

    nodePositions = {};
    dataset.nodes.forEach((node, i) => {
        const angle = (2 * Math.PI * i) / N;
        nodePositions[node.id] = {
            x: centerX + radius * Math.cos(angle),
            y: centerY + radius * Math.sin(angle)
        };
    });

    const edges = [
        ...(dataset.interference_edges || []).map(edge => ({ ...edge, length: 145, strength: 0.002 })),
        ...(dataset.coalescing_edges || []).map(edge => ({ ...edge, length: 95, strength: 0.004 }))
    ];
    const iterations = 100;
    const margin = 28;

    for (let iter = 0; iter < iterations; iter++) {
        const forces = dataset.nodes.map(() => ({ x: 0, y: 0 }));

        for (let i = 0; i < N; i++) {
            for (let j = i + 1; j < N; j++) {
                const p1 = nodePositions[dataset.nodes[i].id];
                const p2 = nodePositions[dataset.nodes[j].id];
                const dx = p2.x - p1.x;
                const dy = p2.y - p1.y;
                const dist = Math.hypot(dx, dy) || 1;
                const magnitude = Math.min(3, Math.max(0, (165 - dist) * 0.035));
                const fx = (dx / dist) * magnitude;
                const fy = (dy / dist) * magnitude;
                forces[i].x -= fx;
                forces[i].y -= fy;
                forces[j].x += fx;
                forces[j].y += fy;
            }
        }

        edges.forEach(edge => {
            const p1 = nodePositions[edge.source];
            const p2 = nodePositions[edge.target];
            if (!p1 || !p2) return;

            const dx = p2.x - p1.x;
            const dy = p2.y - p1.y;
            const dist = Math.hypot(dx, dy) || 1;
            const magnitude = (dist - edge.length) * edge.strength;
            const fx = (dx / dist) * magnitude;
            const fy = (dy / dist) * magnitude;
            const i = nodeIndices.get(edge.source);
            const j = nodeIndices.get(edge.target);
            if (i === undefined || j === undefined) return;
            forces[i].x += fx;
            forces[i].y += fy;
            forces[j].x -= fx;
            forces[j].y -= fy;
        });

        const cooling = 1 - (iter / iterations) * 0.65;
        dataset.nodes.forEach((node, i) => {
            const force = forces[i];
            const magnitude = Math.hypot(force.x, force.y) || 1;
            const step = Math.min(5, magnitude) * cooling / magnitude;
            const position = nodePositions[node.id];
            position.x = Math.max(margin, Math.min(width - margin, position.x + force.x * step));
            position.y = Math.max(margin, Math.min(height - margin, position.y + force.y * step));
        });
    }
}

function resetView() {
    zoom = 1.0;
    panX = 0;
    panY = 0;
    drawGraph();
}

function screenToWorld(sx, sy) {
    return {
        x: (sx - panX) / zoom,
        y: (sy - panY) / zoom
    };
}

function worldToScreen(wx, wy) {
    return {
        x: wx * zoom + panX,
        y: wy * zoom + panY
    };
}

function drawGraph() {
    const canvas = document.getElementById("graph-canvas");
    if (!canvas || !dataset) return;
    const ctx = canvas.getContext("2d");

    const dpr = window.devicePixelRatio || 1;
    const width = parseFloat(canvas.style.width) || (canvas.width / dpr);
    const height = parseFloat(canvas.style.height) || (canvas.height / dpr);

    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // Apply pan & zoom
    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);

    // Build fast attention map if enabled
    const attentionMap = {};
    if (showAttention && dataset.attention_edges) {
        dataset.attention_edges.forEach(ae => {
            const k1 = `${ae.source}-${ae.target}`;
            const k2 = `${ae.target}-${ae.source}`;
            attentionMap[k1] = ae.weight;
            attentionMap[k2] = ae.weight;
        });
    }

    // 1. Draw Interference Edges
    dataset.interference_edges.forEach(edge => {
        const p1 = nodePositions[edge.source];
        const p2 = nodePositions[edge.target];
        if (!p1 || !p2) return;

        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);

        if (showAttention) {
            const w = attentionMap[`${edge.source}-${edge.target}`] || 0.07;
            const alpha = Math.min(1.0, Math.max(0.15, w * 4.5));
            const lineWidth = Math.max(1.0, w * 18);
            ctx.strokeStyle = `rgba(168, 85, 247, ${alpha})`;
            ctx.lineWidth = lineWidth;
        } else {
            ctx.strokeStyle = "rgba(239, 68, 68, 0.22)";
            ctx.lineWidth = 1.5;
        }

        ctx.setLineDash([]);
        ctx.stroke();
    });

    // 2. Draw Move Coalescing Edges (Green Dashed)
    dataset.coalescing_edges.forEach(edge => {
        const p1 = nodePositions[edge.source];
        const p2 = nodePositions[edge.target];
        if (!p1 || !p2) return;

        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.strokeStyle = "#10b981";
        ctx.lineWidth = 2.8;
        ctx.setLineDash([6, 5]);
        ctx.stroke();
        ctx.setLineDash([]);
    });

    // 3. Draw Nodes
    dataset.nodes.forEach(node => {
        const pos = nodePositions[node.id];
        if (!pos) return;

        const assignment = getNodeAssignment(node, currentModel);
        const color = getNodeColor(assignment);
        const isHovered = hoverNodeId === node.id;
        const isDisagreement = highlightDisagreements && node.is_disagreement;

        // Disagreement outer beacon ring
        if (isDisagreement) {
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 25, 0, 2 * Math.PI);
            ctx.strokeStyle = "#f59e0b";
            ctx.lineWidth = 3;
            ctx.setLineDash([4, 3]);
            ctx.stroke();
            ctx.setLineDash([]);

            // Amber aura
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 28, 0, 2 * Math.PI);
            ctx.fillStyle = "rgba(245, 158, 11, 0.15)";
            ctx.fill();
        }

        // Soft outer glow
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, isHovered ? 24 : 20, 0, 2 * Math.PI);
        ctx.fillStyle = color + "33";
        ctx.fill();

        // Main node circle
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 16, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();

        // Node border
        ctx.lineWidth = isHovered ? 3.5 : 2;
        ctx.strokeStyle = isHovered ? "#ffffff" : "rgba(255, 255, 255, 0.85)";
        ctx.stroke();

        // Node ID label
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 11px Outfit, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(node.id, pos.x, pos.y);

        // Small register assignment pill badge beneath node
        const badgeY = pos.y + 24;
        ctx.beginPath();
        ctx.roundRect ? ctx.roundRect(pos.x - 14, badgeY - 7, 28, 14, 4) : 
            ctx.rect(pos.x - 14, badgeY - 7, 28, 14);
        ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = color;
        ctx.font = "600 8.5px Fira Code, monospace";
        ctx.fillText(assignment === "SPILL" ? "SP" : assignment, pos.x, badgeY);
    });

    ctx.restore();
}

// -------------------------------------------------------------
// Mouse & Canvas Interaction Listeners
// -------------------------------------------------------------
function setupCanvasInteractions() {
    const canvas = document.getElementById("graph-canvas");
    const tooltip = document.getElementById("node-tooltip");
    if (!canvas || !tooltip) return;

    canvas.addEventListener("mousedown", (e) => {
        if (e.button !== 0) return; // Only left click
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        const world = screenToWorld(sx, sy);

        // Check if mouse clicked on a node
        let clickedNode = null;
        if (dataset && dataset.nodes) {
            for (const n of dataset.nodes) {
                const pos = nodePositions[n.id];
                if (!pos) continue;
                if (Math.hypot(world.x - pos.x, world.y - pos.y) <= 20) {
                    clickedNode = n;
                    break;
                }
            }
        }

        if (clickedNode) {
            isDraggingNode = true;
            draggedNodeId = clickedNode.id;
            hoverNodeId = clickedNode.id;
        } else {
            isPanning = true;
            panStartX = e.clientX - panX;
            panStartY = e.clientY - panY;
        }
    });

    window.addEventListener("mousemove", (e) => {
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;

        if (isDraggingNode && draggedNodeId) {
            const world = screenToWorld(sx, sy);
            if (nodePositions[draggedNodeId]) {
                nodePositions[draggedNodeId].x = world.x;
                nodePositions[draggedNodeId].y = world.y;
                drawGraph();
                updateTooltip(draggedNodeId, sx, sy);
            }
            return;
        }

        if (isPanning) {
            panX = e.clientX - panStartX;
            panY = e.clientY - panStartY;
            drawGraph();
            return;
        }

        // Hover detection if inside canvas bounds
        if (sx >= 0 && sx <= rect.width && sy >= 0 && sy <= rect.height) {
            const world = screenToWorld(sx, sy);
            let found = null;
            if (dataset && dataset.nodes) {
                for (const n of dataset.nodes) {
                    const pos = nodePositions[n.id];
                    if (!pos) continue;
                    if (Math.hypot(world.x - pos.x, world.y - pos.y) <= 20) {
                        found = n;
                        break;
                    }
                }
            }

            if (found) {
                hoverNodeId = found.id;
                updateTooltip(found.id, sx, sy);
                drawGraph();
            } else if (hoverNodeId) {
                hoverNodeId = null;
                tooltip.classList.add("hidden");
                drawGraph();
            }
        }
    });

    window.addEventListener("mouseup", () => {
        isDraggingNode = false;
        draggedNodeId = null;
        isPanning = false;
    });

    // Zoom on mouse wheel centered on mouse pointer
    canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;

        const zoomFactor = e.deltaY < 0 ? 1.12 : 0.89;
        const newZoom = Math.min(3.0, Math.max(0.35, zoom * zoomFactor));

        // Keep mouse point stationary in world space
        panX = sx - (sx - panX) * (newZoom / zoom);
        panY = sy - (sy - panY) * (newZoom / zoom);
        zoom = newZoom;

        drawGraph();
    }, { passive: false });
}

function updateTooltip(nodeId, sx, sy) {
    const tooltip = document.getElementById("node-tooltip");
    if (!tooltip || !dataset) return;

    const node = dataset.nodes.find(n => n.id === nodeId);
    if (!node) return;

    const asgn = getNodeAssignment(node, currentModel);
    const color = getNodeColor(asgn);

    let probHtml = "";
    const spillProbability = getNodeModelValue(node, "spill_probs", currentModel);
    if (spillProbability !== undefined) {
        probHtml = `<br/>Spill Probability: <strong>${(spillProbability * 100).toFixed(1)}%</strong>`;
    }

    let confHtml = "";
    const confidence = getNodeModelValue(node, "confidence", currentModel);
    if (confidence !== undefined) {
        confHtml = `<br/>Confidence: <strong>${(confidence * 100).toFixed(1)}%</strong>`;
    }

    let disagreeHtml = "";
    if (node.is_disagreement && node.assignments) {
        const votes = Object.entries(node.assignments)
            .map(([m, a]) => `${m.toUpperCase()}: ${a}`)
            .join(" | ");
        disagreeHtml = `<div style="margin-top:6px; padding-top:6px; border-top:1px solid rgba(255,255,255,0.15); font-size:0.75rem;">
            <strong style="color:#f59e0b;">Model Decisions:</strong><br/>
            ${votes}
        </div>`;
    }

    tooltip.innerHTML = `
        <strong>Variable ${node.id}</strong><br/>
        Loop Depth: ${node.loop_depth !== undefined ? node.loop_depth : 0}<br/>
        Spill Cost: ${(node.spill_cost || 0).toFixed(1)} | Degree: ${node.degree || 0}<br/>
        Active Model (${currentModel}): <span style="color:${color}; font-weight:700;">${asgn}</span>
        ${probHtml}
        ${confHtml}
        ${disagreeHtml}
    `;

    tooltip.classList.remove("hidden");
    tooltip.style.left = `${sx + 15}px`;
    tooltip.style.top = `${sy + 15}px`;
}

// -------------------------------------------------------------
// Fallback Sample Data if backend is offline
// -------------------------------------------------------------
function getFallbackData() {
    return {
        "program_name": "func_demo_fallback",
        "num_registers": 4,
        "instructions": [
            "L00: v0 = const_1",
            "L01: v1 = const_5",
            "L02: v2 = v0 + v1",
            "L03: MOVE v3 <- v2",
            "L04: v4 = v3 * v1",
            "L05: RETURN v4"
        ],
        "cfg_blocks": [
            {
                "block_id": 0,
                "label": null,
                "loop_depth": 0,
                "instructions": ["v0 = const_1", "v1 = const_5", "v2 = v0 + v1", "MOVE v3 <- v2", "v4 = v3 * v1", "RETURN v4"],
                "successors": []
            }
        ],
        "nodes": [
            { "id": "v0", "spill_cost": 1.0, "loop_depth": 0, "degree": 1, "move_degree": 0, "assignments": { "consensus": "R0", "rgcn": "R0", "gat": "R0", "sage": "R0", "gcn": "R0", "cb": "R0" }, "is_disagreement": false },
            { "id": "v1", "spill_cost": 1.0, "loop_depth": 0, "degree": 2, "move_degree": 0, "assignments": { "consensus": "R1", "rgcn": "R1", "gat": "R1", "sage": "R1", "gcn": "R1", "cb": "R1" }, "is_disagreement": false },
            { "id": "v2", "spill_cost": 1.0, "loop_depth": 0, "degree": 1, "move_degree": 1, "assignments": { "consensus": "R2", "rgcn": "R2", "gat": "R2", "sage": "R2", "gcn": "R2", "cb": "R2" }, "is_disagreement": false },
            { "id": "v3", "spill_cost": 1.0, "loop_depth": 0, "degree": 1, "move_degree": 1, "assignments": { "consensus": "R2", "rgcn": "R2", "gat": "R2", "sage": "R2", "gcn": "R2", "cb": "R2" }, "is_disagreement": false },
            { "id": "v4", "spill_cost": 1.0, "loop_depth": 0, "degree": 0, "move_degree": 0, "assignments": { "consensus": "R0", "rgcn": "R0", "gat": "R0", "sage": "R0", "gcn": "R0", "cb": "R0" }, "is_disagreement": false }
        ],
        "interference_edges": [
            { "source": "v0", "target": "v1", "type": "interference" },
            { "source": "v1", "target": "v2", "type": "interference" }
        ],
        "coalescing_edges": [
            { "source": "v2", "target": "v3", "type": "coalescing" }
        ],
        "attention_edges": [
            { "source": "v0", "target": "v1", "weight": 0.5 },
            { "source": "v1", "target": "v2", "weight": 0.5 }
        ],
        "models": ["consensus", "rgcn", "gat", "sage", "gcn", "cb"],
        "model_metadata": {
            "consensus": { "name": "Consensus Ensemble", "specialty": "Confidence-weighted soft voting across 4 agents" },
            "rgcn": { "name": "Relational Agent", "specialty": "Move-chains and coalescing edges" },
            "gat": { "name": "Attention Agent", "specialty": "Top-attended interfering neighbors" },
            "sage": { "name": "Neighbourhood Agent", "specialty": "Hub nodes and dense subgraphs" },
            "gcn": { "name": "Pressure Agent", "specialty": "Global pressure peaks & loop-hot nodes" },
            "cb": { "name": "Chaitin-Briggs", "specialty": "Greedy cost/degree classical allocator" }
        },
        "model_stats": {
            "consensus": { "spills": 0, "spill_cost": 0.0, "moves_eliminated": 1 },
            "rgcn": { "spills": 0, "spill_cost": 0.0, "moves_eliminated": 1 },
            "gat": { "spills": 0, "spill_cost": 0.0, "moves_eliminated": 1 },
            "sage": { "spills": 0, "spill_cost": 0.0, "moves_eliminated": 1 },
            "gcn": { "spills": 0, "spill_cost": 0.0, "moves_eliminated": 1 },
            "cb": { "spills": 0, "spill_cost": 0.0, "moves_eliminated": 1 }
        },
        "disagreement_nodes": [],
        "pairwise_agreement": {
            "rgcn_vs_gat": 100.0,
            "rgcn_vs_sage": 100.0,
            "rgcn_vs_gcn": 100.0,
            "gat_vs_sage": 100.0,
            "gat_vs_gcn": 100.0,
            "sage_vs_gcn": 100.0
        },
        "unanimous_pct": 100.0,
        "explanations": {}
    };
}
