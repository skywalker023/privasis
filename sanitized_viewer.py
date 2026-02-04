#!/usr/bin/env python3
"""
Sanitized Record Viewer Server

A web server to visualize sanitized JSONL outputs from sanitize.py.
Run with: python sanitized_viewer.py --port 8081
Access at: http://<your-ip>:8081

Features:
- Side-by-side comparison of original vs sanitized records
- Visualization of sanitization strategies (drop, abstract, keep)
- Sequence-level diff highlighting
- Attribute-level breakdown

No external dependencies required - uses Python's built-in http.server.
"""

import argparse
import json
import os
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Configuration
OUTPUT_DIR = Path(__file__).parent / "outputs" / "sanitized_privasis"

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sanitized Record Viewer</title>
    <style>
        :root {
            --bg-primary: #0f0f0f;
            --bg-secondary: #1a1a1a;
            --bg-tertiary: #252525;
            --text-primary: #f0f0f0;
            --text-secondary: #a0a0a0;
            --accent: #6366f1;
            --accent-hover: #818cf8;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --info: #3b82f6;
            --border: #333;
            --drop-bg: rgba(239, 68, 68, 0.15);
            --drop-border: #ef4444;
            --abstract-bg: rgba(245, 158, 11, 0.15);
            --abstract-border: #f59e0b;
            --keep-bg: rgba(34, 197, 94, 0.15);
            --keep-border: #22c55e;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 30px;
            flex-wrap: wrap;
            gap: 15px;
        }

        h1 {
            font-size: 1.8rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        h1 span {
            color: var(--accent);
        }

        .file-selector {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }

        select {
            padding: 10px 15px;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--bg-secondary);
            color: var(--text-primary);
            font-size: 0.9rem;
            min-width: 300px;
            cursor: pointer;
        }

        select:focus {
            outline: none;
            border-color: var(--accent);
        }

        .btn {
            background: var(--accent);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 500;
            transition: all 0.2s;
        }

        .btn:hover {
            background: var(--accent-hover);
            transform: translateY(-1px);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .btn-secondary {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
        }

        .btn-secondary:hover {
            background: var(--bg-secondary);
        }

        .navigation {
            display: flex;
            align-items: center;
            gap: 15px;
            background: var(--bg-secondary);
            padding: 15px 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            flex-wrap: wrap;
        }

        .navigation span {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }

        .navigation strong {
            color: var(--text-primary);
        }

        .nav-input {
            width: 80px;
            padding: 8px 12px;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            text-align: center;
            font-size: 0.95rem;
        }

        .stats {
            display: flex;
            gap: 20px;
            margin-left: auto;
            flex-wrap: wrap;
        }

        .stat-item {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .stat-value {
            font-size: 1.2rem;
            font-weight: 600;
        }

        .stat-value.drop { color: var(--error); }
        .stat-value.abstract { color: var(--warning); }
        .stat-value.keep { color: var(--success); }
        .stat-value.total { color: var(--accent); }

        .stat-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
        }

        .empty-state {
            text-align: center;
            padding: 100px 20px;
            color: var(--text-secondary);
        }

        .empty-state svg {
            width: 80px;
            height: 80px;
            margin-bottom: 20px;
            opacity: 0.5;
        }

        .card {
            background: var(--bg-secondary);
            border-radius: 16px;
            overflow: hidden;
            margin-bottom: 25px;
            border: 1px solid var(--border);
        }

        .card-header {
            background: var(--bg-tertiary);
            padding: 15px 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
        }

        .card-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }

        .badge-drop {
            background: var(--drop-bg);
            color: var(--error);
            border: 1px solid var(--drop-border);
        }

        .badge-abstract {
            background: var(--abstract-bg);
            color: var(--warning);
            border: 1px solid var(--abstract-border);
        }

        .badge-keep {
            background: var(--keep-bg);
            color: var(--success);
            border: 1px solid var(--keep-border);
        }

        .badge-info {
            background: rgba(59, 130, 246, 0.15);
            color: var(--info);
            border: 1px solid var(--info);
        }

        .card-content {
            padding: 25px;
        }

        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        @media (max-width: 1200px) {
            .comparison-grid {
                grid-template-columns: 1fr;
            }
        }

        .comparison-panel {
            background: var(--bg-tertiary);
            border-radius: 10px;
            overflow: hidden;
        }

        .panel-header {
            padding: 12px 18px;
            font-weight: 600;
            font-size: 0.9rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .panel-header.original {
            background: rgba(99, 102, 241, 0.15);
            border-bottom: 2px solid var(--accent);
        }

        .panel-header.sanitized {
            background: rgba(34, 197, 94, 0.15);
            border-bottom: 2px solid var(--success);
        }

        .panel-content {
            padding: 20px;
            white-space: pre-wrap;
            font-size: 0.9rem;
            line-height: 1.8;
            max-height: 500px;
            overflow-y: auto;
            font-family: 'Consolas', 'Monaco', monospace;
        }

        .section {
            margin-bottom: 25px;
        }

        .section:last-child {
            margin-bottom: 0;
        }

        .section-title {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            color: var(--accent);
            margin-bottom: 15px;
            letter-spacing: 0.5px;
        }

        .strategy-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 12px;
        }

        .strategy-item {
            background: var(--bg-tertiary);
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid var(--border);
        }

        .strategy-item.drop {
            border-left-color: var(--error);
        }

        .strategy-item.abstract {
            border-left-color: var(--warning);
        }

        .strategy-item.keep {
            border-left-color: var(--success);
        }

        .strategy-attr {
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--text-primary);
        }

        .strategy-method {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        .sequence-mapping {
            background: var(--bg-tertiary);
            border-radius: 10px;
            margin-bottom: 15px;
            overflow: hidden;
        }

        .sequence-header {
            padding: 12px 18px;
            background: rgba(99, 102, 241, 0.1);
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }

        .sequence-header:hover {
            background: rgba(99, 102, 241, 0.15);
        }

        .sequence-idx {
            font-weight: 600;
            color: var(--accent);
        }

        .sequence-attrs {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .sequence-content {
            display: none;
            padding: 18px;
        }

        .sequence-content.expanded {
            display: block;
        }

        .sequence-row {
            display: grid;
            grid-template-columns: 100px 1fr;
            gap: 15px;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--border);
        }

        .sequence-row:last-child {
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }

        .sequence-label {
            font-size: 0.8rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            padding-top: 4px;
        }

        .sequence-text {
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9rem;
            line-height: 1.6;
            background: var(--bg-secondary);
            padding: 12px;
            border-radius: 6px;
        }

        .sequence-text.original {
            border-left: 3px solid var(--accent);
        }

        .sequence-text.sanitized {
            border-left: 3px solid var(--success);
        }

        .target-attrs-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .target-attr-tag {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
        }

        .target-attr-tag.drop {
            background: var(--drop-bg);
            border-color: var(--drop-border);
            color: var(--error);
        }

        .target-attr-tag.abstract {
            background: var(--abstract-bg);
            border-color: var(--abstract-border);
            color: var(--warning);
        }

        .target-attr-tag.keep {
            background: var(--keep-bg);
            border-color: var(--keep-border);
            color: var(--success);
        }

        .profile-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 12px;
        }

        .profile-item {
            background: var(--bg-tertiary);
            padding: 12px 15px;
            border-radius: 8px;
        }

        .profile-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            margin-bottom: 4px;
        }

        .profile-value {
            font-size: 0.95rem;
            color: var(--text-primary);
            word-break: break-word;
        }

        .instruction-box {
            background: var(--bg-tertiary);
            padding: 20px;
            border-radius: 10px;
            white-space: pre-wrap;
            font-size: 0.9rem;
            line-height: 1.7;
            max-height: 400px;
            overflow-y: auto;
        }

        .collapse-btn {
            background: none;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 1.2rem;
            transition: transform 0.2s;
        }

        .collapse-btn.expanded {
            transform: rotate(180deg);
        }

        .tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
            background: var(--bg-tertiary);
            padding: 5px;
            border-radius: 10px;
            width: fit-content;
            flex-wrap: wrap;
        }

        .tab {
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
            border: none;
            background: transparent;
            color: var(--text-secondary);
        }

        .tab:hover {
            background: var(--bg-secondary);
        }

        .tab.active {
            background: var(--accent);
            color: white;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }

        .search-input {
            flex: 1;
            padding: 12px 18px;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--bg-secondary);
            color: var(--text-primary);
            font-size: 0.95rem;
        }

        .search-input::placeholder {
            color: var(--text-secondary);
        }

        .highlight {
            background: rgba(245, 158, 11, 0.3);
            padding: 2px 4px;
            border-radius: 3px;
        }

        .diff-added {
            background: rgba(34, 197, 94, 0.2);
            text-decoration: none;
        }

        .diff-removed {
            background: rgba(239, 68, 68, 0.2);
            text-decoration: line-through;
        }

        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-tertiary);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-secondary);
        }

        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 50px;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .legend {
            display: flex;
            gap: 20px;
            padding: 10px 0;
            flex-wrap: wrap;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        .legend-color {
            width: 16px;
            height: 16px;
            border-radius: 4px;
        }

        .legend-color.drop {
            background: var(--drop-bg);
            border: 1px solid var(--drop-border);
        }

        .legend-color.abstract {
            background: var(--abstract-bg);
            border: 1px solid var(--abstract-border);
        }

        .legend-color.keep {
            background: var(--keep-bg);
            border: 1px solid var(--keep-border);
        }

        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }

            header {
                flex-direction: column;
                align-items: flex-start;
            }

            .navigation {
                flex-direction: column;
                align-items: flex-start;
            }

            .stats {
                margin-left: 0;
                width: 100%;
                justify-content: space-around;
            }

            .profile-grid {
                grid-template-columns: 1fr;
            }

            select {
                min-width: 200px;
                width: 100%;
            }

            .strategy-grid {
                grid-template-columns: 1fr;
            }
        }

        .error-card {
            border-color: var(--error);
        }

        .error-card .card-header {
            background: rgba(239, 68, 68, 0.1);
        }

        .error-message {
            background: var(--drop-bg);
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid var(--error);
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9rem;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Sanitized <span>Record</span> Viewer</h1>
            <div class="file-selector">
                <select id="fileSelect">
                    <option value="">-- Select a file --</option>
                </select>
                <button class="btn" onclick="loadSelectedFile()">Load</button>
                <button class="btn btn-secondary" onclick="refreshFiles()">Refresh</button>
            </div>
        </header>

        <div id="searchContainer" style="display: none;">
            <div class="search-box">
                <input type="text" class="search-input" id="searchInput" placeholder="Search in records...">
                <button class="btn btn-secondary" onclick="clearSearch()">Clear</button>
            </div>
        </div>

        <div id="navigationContainer" style="display: none;">
            <div class="navigation">
                <button class="btn btn-secondary" onclick="navigate(-1)" id="prevBtn">← Previous</button>
                <span>
                    <strong id="currentIndex">1</strong> of <strong id="totalCount">0</strong>
                </span>
                <input type="number" class="nav-input" id="goToInput" min="1" placeholder="Go to">
                <button class="btn btn-secondary" onclick="goToIndex()">Go</button>
                <button class="btn btn-secondary" onclick="navigate(1)" id="nextBtn">Next →</button>
                
                <div class="stats">
                    <div class="stat-item">
                        <span class="stat-value drop" id="dropCount">0</span>
                        <span class="stat-label">Dropped</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value abstract" id="abstractCount">0</span>
                        <span class="stat-label">Abstracted</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value keep" id="keepCount">0</span>
                        <span class="stat-label">Kept</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value total" id="totalRecords">0</span>
                        <span class="stat-label">Records</span>
                    </div>
                </div>
            </div>
        </div>

        <div id="emptyState" class="empty-state">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <h2>No Data Loaded</h2>
            <p>Select a sanitized JSONL file from the dropdown to view results</p>
        </div>

        <div id="contentContainer"></div>
    </div>

    <script>
        let data = [];
        let currentIndex = 0;
        let searchTerm = '';
        let currentFile = '';

        document.addEventListener('DOMContentLoaded', () => {
            refreshFiles();
            document.getElementById('searchInput').addEventListener('input', debounce(handleSearch, 300));
            document.getElementById('goToInput').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') goToIndex();
            });
            
            document.addEventListener('keydown', (e) => {
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
                if (e.key === 'ArrowLeft') navigate(-1);
                if (e.key === 'ArrowRight') navigate(1);
            });
        });

        async function refreshFiles() {
            try {
                const response = await fetch('/api/files');
                const files = await response.json();
                const select = document.getElementById('fileSelect');
                const currentValue = select.value;
                
                select.innerHTML = '<option value="">-- Select a file --</option>';
                files.forEach(file => {
                    const option = document.createElement('option');
                    option.value = file.path;
                    option.textContent = `${file.name} (${file.lines} lines)`;
                    select.appendChild(option);
                });
                
                if (currentValue && files.some(f => f.path === currentValue)) {
                    select.value = currentValue;
                }
            } catch (err) {
                console.error('Error fetching files:', err);
            }
        }

        async function loadSelectedFile() {
            const select = document.getElementById('fileSelect');
            const filePath = select.value;
            if (!filePath) {
                alert('Please select a file first');
                return;
            }

            document.getElementById('contentContainer').innerHTML = '<div class="loading"><div class="spinner"></div></div>';
            document.getElementById('emptyState').style.display = 'none';

            try {
                const response = await fetch(`/api/load?file=${encodeURIComponent(filePath)}`);
                if (!response.ok) {
                    throw new Error(`HTTP error: ${response.status}`);
                }
                data = await response.json();
                currentFile = filePath;
                currentIndex = 0;
                updateUI();
            } catch (err) {
                alert('Error loading file: ' + err.message);
                document.getElementById('contentContainer').innerHTML = '';
                document.getElementById('emptyState').style.display = 'block';
            }
        }

        function updateUI() {
            if (data.length === 0) {
                document.getElementById('emptyState').style.display = 'block';
                document.getElementById('navigationContainer').style.display = 'none';
                document.getElementById('searchContainer').style.display = 'none';
                document.getElementById('contentContainer').innerHTML = '';
                return;
            }

            document.getElementById('emptyState').style.display = 'none';
            document.getElementById('navigationContainer').style.display = 'block';
            document.getElementById('searchContainer').style.display = 'block';
            
            document.getElementById('currentIndex').textContent = currentIndex + 1;
            document.getElementById('totalCount').textContent = data.length;
            document.getElementById('goToInput').max = data.length;
            
            document.getElementById('prevBtn').disabled = currentIndex === 0;
            document.getElementById('nextBtn').disabled = currentIndex === data.length - 1;

            // Calculate stats
            let dropCount = 0, abstractCount = 0, keepCount = 0;
            data.forEach(item => {
                if (item.sanitized_record && item.sanitized_record.sanitization_info) {
                    const info = item.sanitized_record.sanitization_info;
                    if (info.drop_targets) dropCount += Object.keys(info.drop_targets).length;
                    if (info.abstract_targets) abstractCount += Object.keys(info.abstract_targets).length;
                    if (info.keep_targets) keepCount += Object.keys(info.keep_targets).length;
                }
            });
            
            document.getElementById('dropCount').textContent = dropCount;
            document.getElementById('abstractCount').textContent = abstractCount;
            document.getElementById('keepCount').textContent = keepCount;
            document.getElementById('totalRecords').textContent = data.length;

            renderItem(data[currentIndex]);
        }

        function renderItem(item) {
            const container = document.getElementById('contentContainer');
            
            // Check if this is an error entry
            if (item.error_message) {
                container.innerHTML = renderError(item);
                return;
            }
            
            let html = '';

            // Instructions card (at the top)
            if (item.base_instruction || item.smoothed_instruction) {
                html += renderInstructionsCard(item);
            }

            // Main comparison card
            html += renderComparisonCard(item);

            // Sanitization strategies card
            if (item.sanitized_record && item.sanitized_record.sanitization_info) {
                html += renderStrategiesCard(item.sanitized_record.sanitization_info);
            }

            // Sequence mappings card
            if (item.sanitized_record && item.sanitized_record.sequence_sanitization_mapping) {
                html += renderSequenceMappingsCard(item.sanitized_record.sequence_sanitization_mapping);
            }

            // Profile and context card
            if (item.profile || item.situation) {
                html += renderContextCard(item);
            }

            container.innerHTML = html;
            attachEventListeners();
        }

        function renderError(item) {
            return `
                <div class="card error-card">
                    <div class="card-header">
                        <span class="card-title">Error Entry</span>
                        <span class="card-badge badge-drop">Error</span>
                    </div>
                    <div class="card-content">
                        <div class="section">
                            <div class="section-title">Error Message</div>
                            <div class="error-message">${escapeHtml(item.error_message || 'Unknown error')}</div>
                        </div>
                        ${item.traceback ? `
                        <div class="section">
                            <div class="section-title">Traceback</div>
                            <div class="error-message">${escapeHtml(item.traceback)}</div>
                        </div>
                        ` : ''}
                        ${item.record ? `
                        <div class="section">
                            <div class="section-title">Original Record</div>
                            <div class="panel-content">${highlightText(escapeHtml(item.record.text || ''))}</div>
                        </div>
                        ` : ''}
                    </div>
                </div>
            `;
        }

        function renderComparisonCard(item) {
            const originalText = item.record?.text || '';
            const sanitizedText = item.sanitized_record?.text || '';
            const folderId = item.folder_num || item.id || 'N/A';

            return `
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Record Comparison</span>
                        <span class="card-badge badge-info">ID: ${escapeHtml(String(folderId).substring(0, 12))}</span>
                    </div>
                    <div class="card-content">
                        <div class="legend">
                            <div class="legend-item">
                                <div class="legend-color drop"></div>
                                <span>Dropped</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color abstract"></div>
                                <span>Abstracted</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color keep"></div>
                                <span>Kept</span>
                            </div>
                        </div>
                        <div class="comparison-grid">
                            <div class="comparison-panel">
                                <div class="panel-header original">
                                    <span>Original Record</span>
                                    <span class="card-badge badge-info">${originalText.length} chars</span>
                                </div>
                                <div class="panel-content">${highlightText(escapeHtml(originalText))}</div>
                            </div>
                            <div class="comparison-panel">
                                <div class="panel-header sanitized">
                                    <span>Sanitized Record</span>
                                    <span class="card-badge badge-info">${sanitizedText.length} chars</span>
                                </div>
                                <div class="panel-content">${highlightText(escapeHtml(sanitizedText))}</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        function renderStrategiesCard(sanitizationInfo) {
            const strategies = sanitizationInfo.strategies || {};
            const dropTargets = sanitizationInfo.drop_targets || {};
            const abstractTargets = sanitizationInfo.abstract_targets || {};
            const keepTargets = sanitizationInfo.keep_targets || sanitizationInfo.attrs_to_keep || {};

            let html = `
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Sanitization Strategies</span>
                    </div>
                    <div class="card-content">
            `;

            // Target attributes summary
            html += `<div class="section">
                <div class="section-title">Target Attributes</div>
                <div class="target-attrs-grid">`;
            
            for (const attr of Object.keys(dropTargets)) {
                html += `<span class="target-attr-tag drop">${escapeHtml(attr)}</span>`;
            }
            for (const attr of Object.keys(abstractTargets)) {
                html += `<span class="target-attr-tag abstract">${escapeHtml(attr)}</span>`;
            }
            for (const attr of Object.keys(keepTargets)) {
                html += `<span class="target-attr-tag keep">${escapeHtml(attr)}</span>`;
            }
            
            html += `</div></div>`;

            // Detailed strategies
            if (Object.keys(strategies).length > 0) {
                html += `<div class="section">
                    <div class="section-title">Strategy Details</div>
                    <div class="strategy-grid">`;
                
                for (const [attr, strategy] of Object.entries(strategies)) {
                    const type = strategy.toLowerCase().includes('drop') ? 'drop' : 
                                 strategy.toLowerCase().includes('keep') ? 'keep' : 'abstract';
                    html += `
                        <div class="strategy-item ${type}">
                            <div class="strategy-attr">${escapeHtml(attr)}</div>
                            <div class="strategy-method">${highlightText(escapeHtml(strategy))}</div>
                        </div>
                    `;
                }
                
                html += `</div></div>`;
            }

            html += `</div></div>`;
            return html;
        }

        function renderSequenceMappingsCard(mappings) {
            if (!mappings || Object.keys(mappings).length === 0) {
                return '';
            }

            let html = `
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Sequence Sanitization Mappings</span>
                        <span class="card-badge badge-info">${Object.keys(mappings).length} sequences</span>
                    </div>
                    <div class="card-content">
            `;

            for (const [idx, mapping] of Object.entries(mappings)) {
                const attrs = mapping.target_attributes || [];
                const strategies = mapping.strategies_used || [];
                
                html += `
                    <div class="sequence-mapping">
                        <div class="sequence-header" onclick="toggleSequence(this)">
                            <div>
                                <span class="sequence-idx">Sequence #${idx}</span>
                                <span class="sequence-attrs">
                                    ${attrs.map(attr => `<span class="target-attr-tag">${escapeHtml(attr)}</span>`).join('')}
                                </span>
                            </div>
                            <button class="collapse-btn">▼</button>
                        </div>
                        <div class="sequence-content">
                            <div class="sequence-row">
                                <div class="sequence-label">Original</div>
                                <div class="sequence-text original">${highlightText(escapeHtml(mapping.original_sequence || ''))}</div>
                            </div>
                            <div class="sequence-row">
                                <div class="sequence-label">Sanitized</div>
                                <div class="sequence-text sanitized">${highlightText(escapeHtml(mapping.sanitized_sequence || ''))}</div>
                            </div>
                            ${strategies.length > 0 ? `
                            <div class="sequence-row">
                                <div class="sequence-label">Strategies</div>
                                <div>${strategies.map(s => `<span class="target-attr-tag">${escapeHtml(s)}</span>`).join(' ')}</div>
                            </div>
                            ` : ''}
                        </div>
                    </div>
                `;
            }

            html += `</div></div>`;
            return html;
        }

        function renderContextCard(item) {
            let html = `
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Context Information</span>
                    </div>
                    <div class="card-content">
            `;

            // Profile section
            if (item.profile && item.profile.profile) {
                html += `<div class="section">
                    <div class="section-title">Profile</div>
                    <div class="profile-grid">`;
                
                for (const [key, value] of Object.entries(item.profile.profile)) {
                    if (typeof value !== 'object') {
                        html += `
                            <div class="profile-item">
                                <div class="profile-label">${formatKey(key)}</div>
                                <div class="profile-value">${highlightText(escapeHtml(String(value)))}</div>
                            </div>
                        `;
                    }
                }
                
                html += `</div></div>`;
            }

            // Situation section
            if (item.situation && item.situation.text) {
                html += `
                    <div class="section">
                        <div class="section-title">Situation</div>
                        <div class="instruction-box">${highlightText(escapeHtml(item.situation.text))}</div>
                    </div>
                `;
            }

            // Format section
            if (item.format) {
                html += `
                    <div class="section">
                        <div class="section-title">Format</div>
                        <div class="instruction-box">${highlightText(escapeHtml(item.format))}</div>
                    </div>
                `;
            }

            html += `</div></div>`;
            return html;
        }

        function renderInstructionsCard(item) {
            let html = `
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Sanitization Instructions</span>
                    </div>
                    <div class="card-content">
                        <div class="tabs">
                            <button class="tab active" data-tab="base">Base Instruction</button>
                            <button class="tab" data-tab="smoothed">Smoothed Instruction</button>
                        </div>
                        <div class="tab-content active" data-content="base">
                            <div class="instruction-box">${highlightText(escapeHtml(item.base_instruction || 'N/A'))}</div>
                        </div>
                        <div class="tab-content" data-content="smoothed">
                            <div class="instruction-box">${highlightText(escapeHtml(item.smoothed_instruction || 'N/A'))}</div>
                        </div>
                    </div>
                </div>
            `;
            return html;
        }

        function attachEventListeners() {
            // Tab switching
            document.querySelectorAll('.tab').forEach(tab => {
                tab.addEventListener('click', function() {
                    const tabGroup = this.closest('.card-content');
                    tabGroup.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    this.classList.add('active');
                    
                    const tabName = this.dataset.tab;
                    tabGroup.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    tabGroup.querySelector(`[data-content="${tabName}"]`).classList.add('active');
                });
            });
        }

        function toggleSequence(header) {
            const content = header.nextElementSibling;
            const btn = header.querySelector('.collapse-btn');
            content.classList.toggle('expanded');
            btn.classList.toggle('expanded');
        }

        function navigate(direction) {
            const newIndex = currentIndex + direction;
            if (newIndex >= 0 && newIndex < data.length) {
                currentIndex = newIndex;
                updateUI();
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        }

        function goToIndex() {
            const input = document.getElementById('goToInput');
            const index = parseInt(input.value) - 1;
            if (index >= 0 && index < data.length) {
                currentIndex = index;
                updateUI();
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
            input.value = '';
        }

        function handleSearch(event) {
            searchTerm = event.target.value.toLowerCase();
            renderItem(data[currentIndex]);
        }

        function clearSearch() {
            document.getElementById('searchInput').value = '';
            searchTerm = '';
            renderItem(data[currentIndex]);
        }

        function highlightText(text) {
            if (!searchTerm) return text;
            const regex = new RegExp(`(${escapeRegex(searchTerm)})`, 'gi');
            return text.replace(regex, '<span class="highlight">$1</span>');
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function escapeRegex(string) {
            return string.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
        }

        function formatKey(key) {
            return key.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());
        }

        function debounce(func, wait) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        }
    </script>
</body>
</html>
'''


def get_jsonl_files(directory: Path) -> list:
    """Get all JSONL files in the directory and subdirectories."""
    files = []
    if not directory.exists():
        return files
    
    for file_path in directory.rglob("*.jsonl"):
        try:
            with open(file_path, 'r') as f:
                line_count = sum(1 for _ in f)
            
            rel_path = file_path.relative_to(directory)
            files.append({
                "name": str(rel_path),
                "path": str(file_path),
                "lines": line_count
            })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    files.sort(key=lambda x: x["name"])
    return files


def load_jsonl(file_path: str) -> list:
    """Load a JSONL file and return parsed data."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
    return data


class SanitizedViewerHandler(BaseHTTPRequestHandler):
    output_dir = OUTPUT_DIR
    
    def log_message(self, format, *args):
        # Custom log format
        print(f"[{self.log_date_time_string()}] {args[0]}")
    
    def send_json_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def send_html_response(self, html):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        query = urllib.parse.parse_qs(parsed_path.query)
        
        if path == '/' or path == '/index.html':
            self.send_html_response(HTML_TEMPLATE)
        
        elif path == '/api/files':
            files = get_jsonl_files(self.output_dir)
            self.send_json_response(files)
        
        elif path == '/api/load':
            file_path = query.get('file', [None])[0]
            if not file_path:
                self.send_json_response({"error": "No file specified"}, 400)
                return
            
            # Security check
            try:
                resolved_path = Path(file_path).resolve()
                output_resolved = self.output_dir.resolve()
                if not str(resolved_path).startswith(str(output_resolved)):
                    self.send_json_response({"error": "Invalid file path"}, 403)
                    return
            except Exception:
                self.send_json_response({"error": "Invalid file path"}, 400)
                return
            
            if not os.path.exists(file_path):
                self.send_json_response({"error": "File not found"}, 404)
                return
            
            try:
                data = load_jsonl(file_path)
                self.send_json_response(data)
            except Exception as e:
                self.send_json_response({"error": str(e)}, 500)
        
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')


def main():
    parser = argparse.ArgumentParser(description="Sanitized Record Viewer Server")
    parser.add_argument("--port", type=int, default=8081, help="Port to run the server on (default: 8081)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory containing sanitized JSONL files")
    args = parser.parse_args()
    
    if args.output_dir:
        SanitizedViewerHandler.output_dir = Path(args.output_dir)
    
    server_address = (args.host, args.port)
    httpd = HTTPServer(server_address, SanitizedViewerHandler)
    
    # Get the hostname for display
    import socket
    hostname = socket.gethostname()
    try:
        ip_address = socket.gethostbyname(hostname)
    except:
        ip_address = "localhost"
    
    print(f"\n{'='*60}")
    print(f"  Sanitized Record Viewer Server")
    print(f"{'='*60}")
    print(f"  Output directory: {SanitizedViewerHandler.output_dir}")
    print(f"  Hostname: {hostname}")
    print(f"  Local URL: http://localhost:{args.port}")
    print(f"  Network URL: http://{ip_address}:{args.port}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()


if __name__ == "__main__":
    main()
