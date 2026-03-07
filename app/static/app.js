const chatMessages = document.getElementById("chat-messages");
const chatScroll = document.getElementById("chat-scroll");
const queryInput = document.getElementById("query-input");
const sendBtn = document.getElementById("send-btn");
const fileInput = document.getElementById("file-input");
const attachInput = document.getElementById("attach-input");
const uploadFilename = document.getElementById("upload-filename");
const uploadDetails = document.getElementById("upload-details");
const statusText = document.getElementById("status-text");
const chunkCountText = document.getElementById("chunk-count-text");

let totalChunks = 0;
let uploadedFiles = [];

// --- file upload ---

fileInput.addEventListener("change", (e) => handleFiles(e.target.files));
attachInput.addEventListener("change", (e) => handleFiles(e.target.files));

async function handleFiles(files) {
    if (!files.length) return;

    statusText.textContent = "Uploading...";
    uploadFilename.textContent = "Processing...";
    uploadDetails.textContent = "Parsing and embedding chunks";

    const formData = new FormData();
    for (const f of files) {
        formData.append("files", f);
    }

    try {
        const resp = await fetch("/ingest", { method: "POST", body: formData });
        const data = await resp.json();

        totalChunks = data.total_chunks_in_store;
        chunkCountText.textContent = totalChunks + " chunks";

        // show results in upload bar
        const ok = data.ingested.filter((r) => r.status === "ok");
        const skipped = data.ingested.filter((r) => r.status !== "ok");

        if (ok.length > 0) {
            const names = ok.map((r) => r.file);
            uploadedFiles = [...new Set([...uploadedFiles, ...names])];
            uploadFilename.textContent = uploadedFiles.join(", ");

            const totalPages = ok.reduce((s, r) => s + r.pages, 0);
            const totalNewChunks = ok.reduce((s, r) => s + r.chunks, 0);
            uploadDetails.textContent =
                totalPages + " pages · " + totalNewChunks + " chunks · " + totalChunks + " total in store";
        }

        if (skipped.length > 0) {
            const reasons = skipped.map((r) => r.file + ": " + r.reason);
            addBotMessage("Some files had issues:\n" + reasons.join("\n"), []);
        }

        statusText.textContent = "Ready";
    } catch (err) {
        statusText.textContent = "Upload failed";
        uploadFilename.textContent = "Upload error";
        uploadDetails.textContent = err.message;
    }

    // reset inputs so same file can be re-uploaded
    fileInput.value = "";
    attachInput.value = "";
}

// --- chat ---

sendBtn.addEventListener("click", sendMessage);
queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// auto-resize textarea
queryInput.addEventListener("input", () => {
    queryInput.style.height = "auto";
    queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + "px";
});

async function sendMessage() {
    const question = queryInput.value.trim();
    if (!question) return;

    addUserMessage(question);
    queryInput.value = "";
    queryInput.style.height = "auto";

    const loadingEl = addLoadingIndicator();
    statusText.textContent = "Thinking...";

    try {
        const resp = await fetch("/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });
        const data = await resp.json();

        loadingEl.remove();
        addBotMessage(data.answer, data.sources || [], data.hallucination_check);
        statusText.textContent = "Ready";
    } catch (err) {
        loadingEl.remove();
        addBotMessage("Something went wrong: " + err.message, []);
        statusText.textContent = "Error";
    }
}

// --- rendering ---

function addUserMessage(text) {
    const wrapper = document.createElement("div");
    wrapper.className = "flex flex-row-reverse gap-4";
    wrapper.innerHTML = `
        <div class="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary shadow-lg shadow-primary/20 mt-1">
            <span class="material-symbols-outlined text-white text-xl">person</span>
        </div>
        <div class="flex flex-col items-end gap-2 max-w-[85%]">
            <div class="rounded-2xl rounded-tr-none bg-primary p-4 text-white shadow-lg shadow-primary/10 leading-relaxed font-medium">
                ${escapeHtml(text)}
            </div>
        </div>
    `;
    chatMessages.appendChild(wrapper);
    scrollToBottom();
}

function addBotMessage(text, sources, halCheck) {
    const wrapper = document.createElement("div");
    wrapper.className = "flex gap-4";

    let sourcesHtml = "";
    if (sources && sources.length > 0) {
        const sourceItems = sources
            .map((s) => `<span class="text-xs text-slate-600">${escapeHtml(s.source)} p.${s.page}</span>`)
            .join(" · ");

        sourcesHtml = `
            <button onclick="this.nextElementSibling.classList.toggle('hidden')" class="group flex items-center gap-1.5 rounded-full border border-slate-200 bg-white/80 px-3 py-1.5 text-[11px] font-bold text-slate-600 hover:border-primary/30 hover:bg-primary/5 hover:text-primary transition-all shadow-sm">
                <span class="material-symbols-outlined text-sm">menu_book</span>
                SOURCES
            </button>
            <div class="hidden rounded-lg bg-slate-50 border border-slate-100 px-3 py-2 mt-1">
                ${sourceItems}
            </div>
        `;
    }

    let halBadge = "";
    if (halCheck && !halCheck.passed) {
        halBadge = `
            <span class="inline-flex items-center gap-1 rounded-full bg-amber-50 border border-amber-200 px-2 py-0.5 text-[10px] font-bold text-amber-700">
                <span class="material-symbols-outlined text-xs">warning</span>
                Some claims may not be fully supported
            </span>
        `;
    }

    wrapper.innerHTML = `
        <div class="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-white shadow-sm border border-slate-100 mt-1">
            <span class="material-symbols-outlined text-primary text-xl">smart_toy</span>
        </div>
        <div class="flex flex-col gap-3 max-w-[85%]">
            <div class="message-shadow rounded-2xl rounded-tl-none bg-white p-6 text-slate-800 leading-relaxed border border-slate-100/50">
                <div class="bot-content">${renderMarkdown(text)}</div>
                ${halBadge}
            </div>
            <div class="flex flex-wrap gap-2 items-start flex-col">
                ${sourcesHtml}
            </div>
        </div>
    `;
    chatMessages.appendChild(wrapper);
    scrollToBottom();
}

function addLoadingIndicator() {
    const wrapper = document.createElement("div");
    wrapper.className = "flex gap-4";
    wrapper.innerHTML = `
        <div class="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-white shadow-sm border border-slate-100 mt-1">
            <span class="material-symbols-outlined text-primary text-xl">smart_toy</span>
        </div>
        <div class="message-shadow rounded-2xl rounded-tl-none bg-white p-6 border border-slate-100/50">
            <div class="flex items-center gap-2 text-slate-400">
                <div class="flex gap-1">
                    <span class="h-2 w-2 rounded-full bg-slate-300 animate-bounce" style="animation-delay:0ms"></span>
                    <span class="h-2 w-2 rounded-full bg-slate-300 animate-bounce" style="animation-delay:150ms"></span>
                    <span class="h-2 w-2 rounded-full bg-slate-300 animate-bounce" style="animation-delay:300ms"></span>
                </div>
                <span class="text-xs font-medium">Searching documents...</span>
            </div>
        </div>
    `;
    chatMessages.appendChild(wrapper);
    scrollToBottom();
    return wrapper;
}

// --- helpers ---

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatScroll.scrollTop = chatScroll.scrollHeight;
    });
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function renderMarkdown(text) {
    // basic markdown: bold, bullets, newlines, code, citations
    let html = escapeHtml(text);

    // bold **text**
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

    // inline code `text`
    html = html.replace(/`([^`]+)`/g, '<code class="bg-slate-100 px-1 py-0.5 rounded text-sm">$1</code>');

    // citation highlight [Source: ...]
    html = html.replace(
        /\[Source:\s*([^\]]+)\]/g,
        '<span class="inline-flex items-center gap-1 text-xs font-medium text-primary bg-primary/5 border border-primary/10 rounded px-1.5 py-0.5"><span class="material-symbols-outlined text-xs">menu_book</span>$1</span>'
    );

    // warning emoji from hallucination filter
    html = html.replace(
        /⚠️\s*/g,
        '<span class="inline-flex items-center text-amber-500 mr-1"><span class="material-symbols-outlined text-sm">warning</span></span>'
    );

    // bullet lines that start with - or •
    html = html.replace(/^[\-•]\s+(.+)$/gm, '<li class="flex gap-2"><span class="text-primary font-bold">•</span><span>$1</span></li>');

    // numbered lines
    html = html.replace(/^(\d+)\.\s+(.+)$/gm, '<li class="flex gap-2"><span class="text-primary font-bold">$1.</span><span>$2</span></li>');

    // wrap consecutive <li> in <ul>
    html = html.replace(/((?:<li[^>]*>.*<\/li>\n?)+)/g, '<ul class="space-y-1 list-none my-2">$1</ul>');

    // paragraphs from double newlines
    html = html.replace(/\n\n/g, '</p><p class="mt-3">');
    html = html.replace(/\n/g, "<br/>");
    html = '<p>' + html + '</p>';

    return html;
}
