#!/usr/bin/env python3
"""
Generation Viewer Server

A web server to visualize JSONL generation outputs.
Run with: python viewer.py --port 8080
Access at: http://<your-ip>:8080

No external dependencies required - uses Python's built-in http.server.
"""

import argparse
import json
import os
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Configuration
OUTPUT_DIR = Path(__file__).parent / "outputs" / "privasis"

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generation Viewer</title>
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
            --border: #333;
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
            max-width: 1400px;
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

        .file-selector {
            display: flex;
            gap: 10px;
            align-items: center;
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
            color: var(--accent);
        }

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

        .generation-card {
            background: var(--bg-secondary);
            border-radius: 16px;
            overflow: hidden;
            margin-bottom: 25px;
            border: 1px solid var(--border);
        }

        .card-header {
            background: var(--bg-tertiary);
            padding: 20px 25px;
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
            background: var(--accent);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }

        .card-content {
            padding: 25px;
        }

        .section {
            margin-bottom: 30px;
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

        .event-card {
            background: var(--bg-tertiary);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
        }

        .event-title {
            font-weight: 600;
            margin-bottom: 15px;
            color: var(--warning);
        }

        .event-attributes {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .event-attr {
            display: flex;
            gap: 10px;
        }

        .event-attr-name {
            font-weight: 500;
            color: var(--accent);
            min-width: 100px;
            flex-shrink: 0;
        }

        .event-attr-value {
            color: var(--text-secondary);
        }

        .situation-text {
            background: var(--bg-tertiary);
            padding: 20px;
            border-radius: 10px;
            line-height: 1.8;
            color: var(--text-secondary);
            font-style: italic;
        }

        .format {
            background: var(--bg-tertiary);
            padding: 20px;
            border-radius: 10px;
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 0.9rem;
            line-height: 1.6;
        }

        .record-tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
            background: var(--bg-tertiary);
            padding: 5px;
            border-radius: 10px;
            width: fit-content;
            flex-wrap: wrap;
        }

        .record-tab {
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
            border: none;
            background: transparent;
            color: var(--text-secondary);
        }

        .record-tab:hover {
            background: var(--bg-secondary);
        }

        .record-tab.active {
            background: var(--accent);
            color: white;
        }

        .record-tab.accepted {
            border-left: 3px solid var(--success);
        }

        .record-tab.rejected {
            border-left: 3px solid var(--error);
        }

        .record-content {
            background: var(--bg-tertiary);
            padding: 25px;
            border-radius: 10px;
            white-space: pre-wrap;
            font-size: 0.95rem;
            line-height: 1.8;
            max-height: 600px;
            overflow-y: auto;
        }

        .record-content.accepted {
            border-left: 4px solid var(--success);
        }

        .record-content.rejected {
            border-left: 4px solid var(--error);
        }

        .attributes-section {
            margin-top: 25px;
        }

        .attribute-group {
            background: var(--bg-tertiary);
            border-radius: 10px;
            margin-bottom: 15px;
            overflow: hidden;
        }

        .attribute-group-header {
            background: rgba(99, 102, 241, 0.1);
            padding: 12px 18px;
            font-weight: 600;
            font-size: 0.9rem;
            border-bottom: 1px solid var(--border);
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .attribute-group-header:hover {
            background: rgba(99, 102, 241, 0.15);
        }

        .attribute-group-content {
            padding: 15px;
            display: none;
        }

        .attribute-group-content.expanded {
            display: block;
        }

        .attr-item {
            display: flex;
            padding: 8px 0;
            border-bottom: 1px solid var(--border);
        }

        .attr-item:last-child {
            border-bottom: none;
        }

        .attr-key {
            font-weight: 500;
            color: var(--text-secondary);
            min-width: 200px;
            flex-shrink: 0;
        }

        .attr-val {
            color: var(--text-primary);
            word-break: break-word;
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

            .event-attr {
                flex-direction: column;
                gap: 4px;
            }

            .attr-item {
                flex-direction: column;
                gap: 4px;
            }

            .attr-key {
                min-width: auto;
            }

            select {
                min-width: 200px;
                width: 100%;
            }
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

        .file-info {
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Generation Viewer</h1>
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
                        <span class="stat-value" id="profileCount">0</span>
                        <span class="stat-label">Profiles</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value" id="recordCount">0</span>
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
            <p>Select a JSONL file from the dropdown to view generations</p>
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

            let profileCount = 0;
            let recordCount = 0;
            data.forEach(item => {
                if (item.profile) profileCount++;
                if (item.record) recordCount++;
            });
            document.getElementById('profileCount').textContent = profileCount;
            document.getElementById('recordCount').textContent = recordCount;

            renderGeneration(data[currentIndex]);
        }

        function renderGeneration(item) {
            const container = document.getElementById('contentContainer');
            let html = '';

            if (item.profile) {
                html += `
                    <div class="generation-card">
                        <div class="card-header">
                            <span class="card-title">Profile</span>
                            ${item.profile.id ? `<span class="card-badge">ID: ${item.profile.id.substring(0, 8)}...</span>` : ''}
                        </div>
                        <div class="card-content">
                            ${renderProfile(item.profile)}
                        </div>
                    </div>
                `;
            }

            if (item.situation) {
                html += `
                    <div class="generation-card">
                        <div class="card-header">
                            <span class="card-title">Situation</span>
                        </div>
                        <div class="card-content">
                            <div class="section">
                                <div class="section-title">Context</div>
                                <div class="situation-text">${highlightText(escapeHtml(item.situation.text || ''))}</div>
                            </div>
                        </div>
                    </div>
                `;
            }

            if (item.format) {
                html += `
                    <div class="generation-card">
                        <div class="card-header">
                            <span class="card-title">Format</span>
                        </div>
                        <div class="card-content">
                            <div class="format">${highlightText(escapeHtml(item.format))}</div>
                        </div>
                    </div>
                `;
            }

            if (item.record) {
                html += renderRecord(item.record);
            }

            container.innerHTML = html;

            document.querySelectorAll('.record-tab').forEach(tab => {
                tab.addEventListener('click', function() {
                    const tabGroup = this.closest('.section');
                    tabGroup.querySelectorAll('.record-tab').forEach(t => t.classList.remove('active'));
                    this.classList.add('active');
                    
                    const index = this.dataset.index;
                    const type = this.dataset.type;
                    tabGroup.querySelectorAll('.record-panel').forEach(p => p.style.display = 'none');
                    tabGroup.querySelector(`[data-panel="${type}-${index}"]`).style.display = 'block';
                });
            });

            document.querySelectorAll('.attribute-group-header').forEach(header => {
                header.addEventListener('click', function() {
                    const content = this.nextElementSibling;
                    const btn = this.querySelector('.collapse-btn');
                    content.classList.toggle('expanded');
                    btn.classList.toggle('expanded');
                });
            });
        }

        function renderProfile(profile) {
            let html = '';
            
            if (profile.profile) {
                html += '<div class="section"><div class="section-title">Personal Information</div><div class="profile-grid">';
                const mainProfile = profile.profile;
                for (const [key, value] of Object.entries(mainProfile)) {
                    if (typeof value === 'object') continue;
                    html += `
                        <div class="profile-item">
                            <div class="profile-label">${formatKey(key)}</div>
                            <div class="profile-value">${highlightText(escapeHtml(String(value)))}</div>
                        </div>
                    `;
                }
                html += '</div></div>';
            }

            if (profile.event_list && profile.event_list.length > 0) {
                html += '<div class="section"><div class="section-title">Events</div>';
                profile.event_list.forEach(event => {
                    html += `
                        <div class="event-card">
                            <div class="event-title">${highlightText(escapeHtml(event.event || ''))}</div>
                            <div class="event-attributes">
                    `;
                    if (event.attributes) {
                        for (const [key, value] of Object.entries(event.attributes)) {
                            html += `
                                <div class="event-attr">
                                    <span class="event-attr-name">${formatKey(key)}</span>
                                    <span class="event-attr-value">${highlightText(escapeHtml(String(value)))}</span>
                                </div>
                            `;
                        }
                    }
                    html += '</div></div>';
                });
                html += '</div>';
            }

            return html;
        }

        function renderRecord(record) {
            let html = `
                <div class="generation-card">
                    <div class="card-header">
                        <span class="card-title">Generated Record</span>
                        ${record.type ? `<span class="card-badge">${escapeHtml(record.type)}</span>` : ''}
                    </div>
                    <div class="card-content">
            `;

            if (record.text) {
                html += `
                    <div class="section">
                        <div class="section-title">Final Record</div>
                        <div class="record-content">${highlightText(escapeHtml(record.text))}</div>
                    </div>
                `;
            }

            if (record.accepted_records && record.accepted_records.length > 0) {
                html += `
                    <div class="section">
                        <div class="section-title">Accepted Records (${record.accepted_records.length})</div>
                        <div class="record-tabs">
                `;
                record.accepted_records.forEach((_, i) => {
                    html += `<button class="record-tab accepted ${i === 0 ? 'active' : ''}" data-index="${i}" data-type="accepted">Version ${i + 1}</button>`;
                });
                html += '</div>';
                record.accepted_records.forEach((rec, i) => {
                    html += `<div class="record-content accepted record-panel" data-panel="accepted-${i}" style="${i === 0 ? '' : 'display:none;'}">${highlightText(escapeHtml(rec))}</div>`;
                });
                html += '</div>';
            }

            if (record.rejected_records && record.rejected_records.length > 0) {
                html += `
                    <div class="section">
                        <div class="section-title">Rejected Records (${record.rejected_records.length})</div>
                        <div class="record-tabs">
                `;
                record.rejected_records.forEach((_, i) => {
                    html += `<button class="record-tab rejected ${i === 0 ? 'active' : ''}" data-index="${i}" data-type="rejected">Version ${i + 1}</button>`;
                });
                html += '</div>';
                record.rejected_records.forEach((rec, i) => {
                    html += `<div class="record-content rejected record-panel" data-panel="rejected-${i}" style="${i === 0 ? '' : 'display:none;'}">${highlightText(escapeHtml(rec))}</div>`;
                });
                html += '</div>';
            }

            if (record.grouped_attributes) {
                html += '<div class="attributes-section"><div class="section-title">Extracted Attributes</div>';
                for (const [group, attrs] of Object.entries(record.grouped_attributes)) {
                    html += `
                        <div class="attribute-group">
                            <div class="attribute-group-header">
                                ${escapeHtml(group)}
                                <button class="collapse-btn">▼</button>
                            </div>
                            <div class="attribute-group-content">
                    `;
                    for (const [key, value] of Object.entries(attrs)) {
                        html += `
                            <div class="attr-item">
                                <span class="attr-key">${formatKey(key)}</span>
                                <span class="attr-val">${highlightText(escapeHtml(String(value)))}</span>
                            </div>
                        `;
                    }
                    html += '</div></div>';
                }
                html += '</div>';
            }

            html += '</div></div>';
            return html;
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
            renderGeneration(data[currentIndex]);
        }

        function clearSearch() {
            document.getElementById('searchInput').value = '';
            searchTerm = '';
            renderGeneration(data[currentIndex]);
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


class ViewerHandler(BaseHTTPRequestHandler):
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
    parser = argparse.ArgumentParser(description="Generation Viewer Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory containing JSONL files")
    args = parser.parse_args()
    
    if args.output_dir:
        ViewerHandler.output_dir = Path(args.output_dir)
    
    server_address = (args.host, args.port)
    httpd = HTTPServer(server_address, ViewerHandler)
    
    # Get the hostname for display
    import socket
    hostname = socket.gethostname()
    try:
        ip_address = socket.gethostbyname(hostname)
    except:
        ip_address = "localhost"
    
    print(f"\n{'='*60}")
    print(f"  Generation Viewer Server")
    print(f"{'='*60}")
    print(f"  Output directory: {ViewerHandler.output_dir}")
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
