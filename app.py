# app.py
from __future__ import annotations

import os
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from agent import run_advisor
from guardrails import redact_pii

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdvisorIQ")

app = FastAPI(title="AdvisorIQ Terminal", version="2.0")

@app.on_event("startup")
async def startup_event():
    if not os.environ.get("GOOGLE_API_KEY"):
        logger.warning("CRITICAL: GOOGLE_API_KEY environment variable is missing. Ensure it is set before querying.")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AdvisorIQ | Wealth Management & Risk Intelligence</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        darkBg: '#090a0f',
                        lightBg: '#f8fafc',
                        cardDark: '#12151f',
                        cardLight: '#ffffff',
                        borderDark: '#222634',
                        borderLight: '#e2e8f0',
                        accent: '#10b981',
                        accentHover: '#059669',
                    }
                }
            }
        }
    </script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Plus Jakarta Sans', sans-serif; }
        code, pre { font-family: 'JetBrains Mono', monospace; }
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(156, 163, 175, 0.3); border-radius: 3px; }
    </style>
</head>
<body class="bg-lightBg dark:bg-darkBg text-slate-800 dark:text-slate-100 min-h-screen flex flex-col transition-colors duration-300">

    <!-- Top Navigation Bar -->
    <header class="border-b border-borderLight dark:border-borderDark bg-white/70 dark:bg-cardDark/70 backdrop-blur-md sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-8 h-8 rounded-lg bg-emerald-500/10 dark:bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center">
                    <div class="w-3 h-3 bg-emerald-500 rounded-sm"></div>
                </div>
                <span class="font-bold text-lg tracking-tight">Advisor<span class="text-emerald-600 dark:text-emerald-400">IQ</span></span>
                <span class="hidden sm:inline-block text-xs px-2 py-0.5 rounded-full bg-slate-200 dark:bg-slate-800 text-slate-600 dark:text-slate-400 font-medium">v2.0 Terminal</span>
            </div>
            
            <div class="flex items-center space-x-4">
                <button onclick="toggleTheme()" class="p-2 rounded-lg border border-borderLight dark:border-borderDark hover:bg-slate-100 dark:hover:bg-slate-800 transition text-slate-600 dark:text-slate-400" aria-label="Toggle Theme">
                    <svg id="sunIcon" class="w-4 h-4 hidden dark:block" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"></path></svg>
                    <svg id="moonIcon" class="w-4 h-4 block dark:hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"></path></svg>
                </button>
                <div class="flex items-center space-x-2 text-xs text-slate-500 dark:text-slate-400 border-l pl-4 border-borderLight dark:border-borderDark">
                    <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                    <span>System Online</span>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Workspace Container -->
    <main class="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        
        <!-- Left Panel: Input & Controls -->
        <section class="lg:col-span-5 bg-white dark:bg-cardDark border border-borderLight dark:border-borderDark rounded-2xl p-6 shadow-sm flex flex-col space-y-6">
            <div>
                <h1 class="text-xl font-semibold tracking-tight">Intelligence Query</h1>
                <p class="text-xs text-slate-500 dark:text-slate-400 mt-1">Submit advisory questions or upload multiple transaction ledgers for risk screening.</p>
            </div>

            <form id="queryForm" onsubmit="handleSubmit(event)" class="space-y-5">
                <div class="space-y-2">
                    <label for="question" class="block text-xs font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400">Prompt / Question</label>
                    <textarea id="question" rows="4" required placeholder="e.g., Analyze AAPL stock trend or review compliance regulations..." class="w-full bg-slate-50 dark:bg-[#0d1017] border border-borderLight dark:border-borderDark rounded-xl p-3 text-sm focus:outline-none focus:border-emerald-500 dark:focus:border-emerald-500 transition resize-none"></textarea>
                </div>

                <div class="space-y-2">
                    <label class="block text-xs font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400">Transaction Statements (CSV - Multiple)</label>
                    <label class="flex flex-col items-center justify-center border-2 border-dashed border-borderLight dark:border-borderDark rounded-xl p-4 cursor-pointer hover:border-emerald-500 dark:hover:border-emerald-500 transition bg-slate-50/50 dark:bg-[#0d1017]/50">
                        <div class="flex items-center space-x-2 text-xs text-slate-500 dark:text-slate-400">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg>
                            <span>Click to add CSV files</span>
                        </div>
                        <input type="file" id="csvFiles" accept=".csv" multiple class="hidden" onchange="handleFilesSelected(this)">
                    </label>
                    
                    <!-- Selected Files List Container -->
                    <div id="fileListContainer" class="space-y-2 mt-2"></div>
                </div>

                <button type="submit" id="submitBtn" class="w-full py-3 px-4 bg-emerald-600 hover:bg-emerald-500 text-white font-medium rounded-xl text-sm transition shadow-lg shadow-emerald-600/20 flex items-center justify-center space-x-2">
                    <span id="btnText">Run Analysis</span>
                    <svg id="spinner" class="w-4 h-4 animate-spin hidden" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path></svg>
                </button>
            </form>
        </section>

        <!-- Right Panel: Results & Feed -->
        <section class="lg:col-span-7 bg-white dark:bg-cardDark border border-borderLight dark:border-borderDark rounded-2xl p-6 shadow-sm flex flex-col min-h-[500px]">
            <div class="flex items-center justify-between border-b border-borderLight dark:border-borderDark pb-4 mb-4">
                <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">Analysis Feed</h2>
                <span id="statusBadge" class="text-xs px-2.5 py-1 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 font-medium">Idle</span>
            </div>

            <div id="outputContainer" class="flex-1 custom-scrollbar overflow-y-auto text-sm leading-relaxed space-y-4 text-slate-600 dark:text-slate-300">
                <div class="text-slate-400 dark:text-slate-500 italic py-12 text-center">
                    Awaiting prompt input. Results, market metrics, and risk classifications will render here in real-time.
                </div>
            </div>
        </section>

    </main>

    <!-- Footer -->
    <footer class="border-t border-borderLight dark:border-borderDark py-4 text-center text-xs text-slate-400 dark:text-slate-500">
        Enterprise PII Redaction Active • Educational & Regulatory Compliance Framework
    </footer>

    <script>
        let selectedFilesList = [];

        function toggleTheme() {
            const html = document.documentElement;
            if (html.classList.contains('dark')) {
                html.classList.remove('dark');
                localStorage.theme = 'light';
            } else {
                html.classList.add('dark');
                localStorage.theme = 'dark';
            }
        }

        function handleFilesSelected(input) {
            if (input.files) {
                for (let i = 0; i < input.files.length; i++) {
                    selectedFilesList.push(input.files[i]);
                }
                renderFileList();
            }
            input.value = ''; 
        }

        function removeFile(index) {
            selectedFilesList.splice(index, 1);
            renderFileList();
        }

        function renderFileList() {
            const container = document.getElementById('fileListContainer');
            container.innerHTML = '';
            
            selectedFilesList.forEach((file, index) => {
                const item = document.createElement('div');
                item.className = 'flex items-center justify-between bg-slate-50 dark:bg-[#0d1017] border border-borderLight dark:border-borderDark px-3 py-2 rounded-xl text-xs';
                item.innerHTML = `
                    <span class="truncate max-w-[220px] text-slate-600 dark:text-slate-300">${file.name}</span>
                    <button type="button" onclick="removeFile(${index})" class="text-slate-400 hover:text-red-500 transition p-1 font-bold">✕</button>
                `;
                container.appendChild(item);
            });
        }

        async function handleSubmit(event) {
            event.preventDefault();
            const question = document.getElementById('question').value;
            const submitBtn = document.getElementById('submitBtn');
            const btnText = document.getElementById('btnText');
            const spinner = document.getElementById('spinner');
            const statusBadge = document.getElementById('statusBadge');
            const outputContainer = document.getElementById('outputContainer');

            if (!question.trim()) return;

            submitBtn.disabled = true;
            btnText.textContent = 'Processing...';
            spinner.classList.remove('hidden');
            statusBadge.textContent = 'Executing Agent';
            statusBadge.className = 'text-xs px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-medium animate-pulse';

            const formData = new FormData();
            formData.append('question', question);
            
            selectedFilesList.forEach(file => {
                formData.append('csv_files', file);
            });

            try {
                const response = await fetch('/api/advise', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                if (response.ok) {
                    outputContainer.innerHTML = `<div class="p-4 rounded-xl bg-slate-50 dark:bg-[#0d1017] border border-borderLight dark:border-borderDark whitespace-pre-wrap">${data.result}</div>`;
                    statusBadge.textContent = 'Completed';
                    statusBadge.className = 'text-xs px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-medium';
                } else {
                    outputContainer.innerHTML = `<div class="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-500">${data.detail || 'An error occurred during execution.'}</div>`;
                    statusBadge.textContent = 'Failed';
                    statusBadge.className = 'text-xs px-2.5 py-1 rounded-md bg-red-500/10 text-red-500 font-medium';
                }
            } catch (err) {
                outputContainer.innerHTML = `<div class="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-500">Network error: Could not reach backend service.</div>`;
                statusBadge.textContent = 'Error';
                statusBadge.className = 'text-xs px-2.5 py-1 rounded-md bg-red-500/10 text-red-500 font-medium';
            } finally {
                submitBtn.disabled = false;
                btnText.textContent = 'Run Analysis';
                spinner.classList.add('hidden');
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTML_TEMPLATE

@app.post("/api/advise")
async def api_advise(question: str = Form(...), csv_files: list[UploadFile] = File(None)):
    logger.info(f"Incoming query received. Processing prompt length: {len(question)}")
    try:
        csv_path = None
        MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB limit
        
        if csv_files and len(csv_files) > 0 and csv_files[0].filename:
            combined_content = ""
            for file in csv_files:
                if not file.filename.endswith('.csv'):
                    raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not a CSV.")
                
                content = await file.read()
                if len(content) > MAX_FILE_SIZE:
                    raise HTTPException(status_code=400, detail=f"File '{file.filename}' exceeds 5MB limit.")
                
                combined_content += redact_pii(content.decode("utf-8", errors="ignore")) + "\n"
            
            csv_path = os.path.join(tempfile.gettempdir(), "advisoriq_upload.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(combined_content)
        
        result = run_advisor(question, csv_path)
        logger.info("Advisor execution completed successfully.")
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), reload=True)