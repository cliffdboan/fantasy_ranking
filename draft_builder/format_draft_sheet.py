#!/usr/bin/env python3
"""
Fantasy Draft Sheet Manager
Creates draft sheets, analyzes rookies, and manages draft data
"""

import pandas as pd
import numpy as np
import re

DRAFT_YEAR = 2025

def format_draft_sheet():
    # Read the CSV files
    df = pd.read_csv(f'draft_sheet_{DRAFT_YEAR}.csv')
    team_df = pd.read_csv(f'../team_data/{DRAFT_YEAR}/team_context_{DRAFT_YEAR}.csv')
    adp_df = pd.read_csv(f'../FantasyPros_{DRAFT_YEAR}_Overall_ADP_Rankings.csv')

    # Merge team context data
    df = df.merge(team_df[['Team', 'OL_Rank', 'Pass_Attempts_Proj', 'Rush_Attempts_Proj']],
                  on='Team', how='left')

    # Clean and merge ADP data (always update from FantasyPros)
    name_mappings = {
        'Cam Ward': 'Cameron Ward'
    }

    def clean_name(name):
        # Remove suffixes and special characters
        cleaned = str(name).replace('*', '').replace('+', '')
        cleaned = re.sub(r'\s+(Jr\.|Sr\.|III|II)$', '', cleaned)
        # Normalize initials (D.J. -> DJ, A.J. -> AJ, etc.)
        cleaned = re.sub(r'([A-Z])\.([A-Z])\.', r'\1\2', cleaned)
        # Apply name mappings
        if cleaned in name_mappings:
            cleaned = name_mappings[cleaned]
        return cleaned.strip()

    adp_df['Player_Clean'] = adp_df['Player'].apply(clean_name)
    df['Player_Clean'] = df['Player'].apply(clean_name)

    # Merge ADP data (drop existing ADP column if present)
    if 'ADP' in df.columns:
        df = df.drop('ADP', axis=1)

    adp_merge = adp_df[['Player_Clean', 'AVG']].copy()
    adp_merge['AVG'] = pd.to_numeric(adp_merge['AVG'], errors='coerce')
    df = df.merge(adp_merge.rename(columns={'AVG': 'ADP'}), on='Player_Clean', how='left')

    # Ensure ADP is numeric and clean up
    df['ADP'] = pd.to_numeric(df['ADP'], errors='coerce')
    df = df.drop('Player_Clean', axis=1)  # Remove helper column

    # Color scheme for different categories
    colors = {
        'Strong Buy': '#28a745',      # Green
        'Buy': '#6f9654',             # Light green
        'Slight Buy': '#8bc34a',      # Lighter green
        'Consensus': '#6c757d',       # Gray
        'Slight Fade': '#ffc107',     # Yellow
        'Fade': '#fd7e14',            # Orange
        'Strong Fade': '#dc3545'      # Red
    }

    position_colors = {
        'QB': '#e3f2fd',    # Light blue
        'RB': '#e8f5e8',    # Light green
        'WR': '#fff3e0',    # Light orange
        'TE': '#f3e5f5'     # Light purple
    }

    tier_colors = {
        'QB1': '#1976d2',
        'QB2': '#42a5f5',
        'QB3+': '#90caf9',
        'RB1/2': '#2e7d32',
        'RB3': '#66bb6a',
        'RB4+': '#a5d6a7',
        'WR1/2': '#f57c00',
        'WR3': '#ffb74d',
        'WR4+': '#ffcc02',
        'TE2': '#7b1fa2',
        'TE3+': '#ba68c8'
    }

    # Create HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fantasy Draft Sheet 2025</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 10px;
                background-color: #f8f9fa;
            }
            .header {
                text-align: center;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .legend {
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: 10px;
                margin-bottom: 20px;
                padding: 15px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .legend-item {
                padding: 5px 10px;
                border-radius: 15px;
                font-size: 12px;
                font-weight: bold;
                color: white;
                text-shadow: 1px 1px 1px rgba(0,0,0,0.3);
            }
            table {
                width: 100%;
                border-collapse: collapse;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            th {
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white;
                padding: 12px 8px;
                text-align: left;
                font-weight: bold;
                font-size: 13px;
                position: sticky;
                top: 0;
                z-index: 10;
                cursor: pointer;
                user-select: none;
            }
            th:hover {
                background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            }
            th.sort-asc::after {
                content: ' ↑';
                font-size: 12px;
            }
            th.sort-desc::after {
                content: ' ↓';
                font-size: 12px;
            }
            td {
                padding: 8px;
                border-bottom: 1px solid #e9ecef;
                font-size: 12px;
                vertical-align: middle;
            }
            .rank {
                font-weight: bold;
                font-size: 14px;
                text-align: center;
                width: 40px;
            }
            .player {
                font-weight: bold;
                min-width: 150px;
            }
            .position {
                text-align: center;
                font-weight: bold;
                width: 40px;
                border-radius: 4px;
                color: #333;
            }
            .team {
                text-align: center;
                font-size: 11px;
                width: 40px;
            }
            .age {
                text-align: center;
                width: 35px;
            }
            .ol-rank {
                text-align: center;
                width: 35px;
                font-weight: bold;
            }
            .adp {
                text-align: center;
                width: 40px;
                font-weight: bold;
            }
            .points {
                text-align: center;
                font-weight: bold;
                width: 60px;
            }
            .edge {
                text-align: center;
                font-weight: bold;
                width: 50px;
            }
            .edge-category {
                text-align: center;
                font-weight: bold;
                border-radius: 12px;
                color: white;
                text-shadow: 1px 1px 1px rgba(0,0,0,0.3);
                width: 80px;
            }
            .tier {
                text-align: center;
                font-weight: bold;
                color: white;
                border-radius: 4px;
                text-shadow: 1px 1px 1px rgba(0,0,0,0.3);
                width: 60px;
            }
            .notes {
                font-size: 11px;
                max-width: 200px;
                word-wrap: break-word;
            }
            .drafted {
                opacity: 0.4;
                text-decoration: line-through;
            }
            tr:hover {
                background-color: #f8f9fa;
                transform: scale(1.01);
                transition: all 0.2s ease;
            }
            .top-100 {
                border-left: 4px solid #28a745;
            }
            .top-50 {
                border-left: 4px solid #ffc107;
            }
            .top-25 {
                border-left: 4px solid #dc3545;
            }
            .filters {
                background: white;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
                align-items: center;
            }
            .filter-group {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .filter-btn {
                padding: 6px 12px;
                border: 2px solid #ddd;
                background: white;
                border-radius: 20px;
                font-size: 12px;
                cursor: pointer;
                transition: all 0.2s;
            }
            .filter-btn.active {
                background: #007bff;
                color: white;
                border-color: #007bff;
            }
            .search-input {
                padding: 8px 12px;
                border: 2px solid #ddd;
                border-radius: 20px;
                font-size: 14px;
                width: 200px;
            }
        </style>
        <script>
            // Load saved state on page load
            window.addEventListener('load', function() {
                loadSavedState();
            });

            function toggleDrafted(row) {
                row.classList.toggle('drafted');
                saveDraftedState();
            }

            function saveDraftedState() {
                const draftedPlayers = [];
                document.querySelectorAll('tbody tr.drafted').forEach(row => {
                    const playerName = row.children[1].textContent;
                    draftedPlayers.push(playerName);
                });
                localStorage.setItem('draftedPlayers', JSON.stringify(draftedPlayers));
            }

            function loadDraftedState() {
                const saved = localStorage.getItem('draftedPlayers');
                if (saved) {
                    const draftedPlayers = JSON.parse(saved);
                    document.querySelectorAll('tbody tr').forEach(row => {
                        const playerName = row.children[1].textContent;
                        if (draftedPlayers.includes(playerName)) {
                            row.classList.add('drafted');
                        }
                    });
                }
            }

            function filterPosition(pos) {
                const rows = document.querySelectorAll('tbody tr');
                const buttons = document.querySelectorAll('.pos-filter');

                buttons.forEach(btn => btn.classList.remove('active'));
                event.target.classList.add('active');

                rows.forEach(row => {
                    const position = row.children[2].textContent;
                    if (pos === 'ALL' || position === pos) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                });

                localStorage.setItem('positionFilter', pos);
            }

            function filterTier(tier) {
                const rows = document.querySelectorAll('tbody tr');
                const buttons = document.querySelectorAll('.tier-filter');

                buttons.forEach(btn => btn.classList.remove('active'));
                event.target.classList.add('active');

                rows.forEach(row => {
                    const rowTier = row.children[10].textContent;
                    if (tier === 'ALL' || rowTier.includes(tier)) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                });

                localStorage.setItem('tierFilter', tier);
            }

            function searchPlayers() {
                const search = document.getElementById('search').value.toLowerCase();
                const rows = document.querySelectorAll('tbody tr');

                rows.forEach(row => {
                    const name = row.children[1].textContent.toLowerCase();
                    if (name.includes(search)) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                });

                localStorage.setItem('searchTerm', search);
            }

            function filterADP(hasADP) {
                const rows = document.querySelectorAll('tbody tr');
                const buttons = document.querySelectorAll('.adp-filter');

                buttons.forEach(btn => btn.classList.remove('active'));
                event.target.classList.add('active');

                rows.forEach(row => {
                    const adpValue = row.children[7].textContent.trim();
                    const hasAdpData = adpValue !== '-';

                    if (hasADP === 'ALL' || (hasADP === 'HAS_ADP' && hasAdpData) || (hasADP === 'NO_ADP' && !hasAdpData)) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                });

                localStorage.setItem('adpFilter', hasADP);
            }

            function resetFilters() {
                document.getElementById('search').value = '';
                document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
                document.querySelector('.pos-filter').classList.add('active');
                document.querySelector('.tier-filter').classList.add('active');
                document.querySelector('.adp-filter').classList.add('active');
                document.querySelectorAll('tbody tr').forEach(row => {
                    row.style.display = '';
                    row.classList.remove('drafted');
                });

                // Clear localStorage
                localStorage.removeItem('searchTerm');
                localStorage.removeItem('positionFilter');
                localStorage.removeItem('tierFilter');
                localStorage.removeItem('adpFilter');
                localStorage.removeItem('draftedPlayers');
                localStorage.removeItem('sortColumn');
                localStorage.removeItem('sortDirection');
            }

            function sortTable(columnIndex) {
                const table = document.querySelector('table');
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                const header = table.querySelectorAll('th')[columnIndex];

                // Determine sort direction
                const isAsc = header.classList.contains('sort-asc');

                // Clear all sort indicators
                table.querySelectorAll('th').forEach(th => {
                    th.classList.remove('sort-asc', 'sort-desc');
                });

                // Set new sort indicator
                header.classList.add(isAsc ? 'sort-desc' : 'sort-asc');

                // Sort rows
                rows.sort((a, b) => {
                    let aVal = a.children[columnIndex].textContent.trim();
                    let bVal = b.children[columnIndex].textContent.trim();

                    // Handle numeric columns
                    if (columnIndex === 0 || columnIndex === 4 || columnIndex === 5 || columnIndex === 6 || columnIndex === 7 || columnIndex === 8) {
                        aVal = parseFloat(aVal.replace(/[^0-9.-]/g, '')) || 0;
                        bVal = parseFloat(bVal.replace(/[^0-9.-]/g, '')) || 0;
                        return isAsc ? bVal - aVal : aVal - bVal;
                    }

                    // Handle text columns
                    return isAsc ? bVal.localeCompare(aVal) : aVal.localeCompare(bVal);
                });

                // Re-append sorted rows
                rows.forEach(row => tbody.appendChild(row));

                // Save sort state
                localStorage.setItem('sortColumn', columnIndex);
                localStorage.setItem('sortDirection', isAsc ? 'desc' : 'asc');
            }

            function loadSavedState() {
                // Load search term
                const savedSearch = localStorage.getItem('searchTerm');
                if (savedSearch) {
                    document.getElementById('search').value = savedSearch;
                    searchPlayers();
                }

                // Load position filter
                const savedPosition = localStorage.getItem('positionFilter');
                if (savedPosition) {
                    document.querySelectorAll('.pos-filter').forEach(btn => {
                        btn.classList.remove('active');
                        if (btn.textContent === savedPosition) {
                            btn.classList.add('active');
                        }
                    });
                    filterPosition(savedPosition);
                }

                // Load tier filter
                const savedTier = localStorage.getItem('tierFilter');
                if (savedTier) {
                    document.querySelectorAll('.tier-filter').forEach(btn => {
                        btn.classList.remove('active');
                        if (btn.textContent === savedTier) {
                            btn.classList.add('active');
                        }
                    });
                    filterTier(savedTier);
                }

                // Load ADP filter
                const savedADP = localStorage.getItem('adpFilter');
                if (savedADP) {
                    document.querySelectorAll('.adp-filter').forEach(btn => {
                        btn.classList.remove('active');
                        if ((btn.textContent === 'ALL' && savedADP === 'ALL') ||
                            (btn.textContent === 'Has ADP' && savedADP === 'HAS_ADP') ||
                            (btn.textContent === 'No ADP' && savedADP === 'NO_ADP')) {
                            btn.classList.add('active');
                        }
                    });
                    filterADP(savedADP);
                }

                // Load sort state
                const savedColumn = localStorage.getItem('sortColumn');
                const savedDirection = localStorage.getItem('sortDirection');
                if (savedColumn && savedDirection) {
                    const columnIndex = parseInt(savedColumn);
                    const header = document.querySelectorAll('th')[columnIndex];
                    header.classList.add(savedDirection === 'asc' ? 'sort-asc' : 'sort-desc');
                    sortTable(columnIndex);
                }

                // Load drafted players (must be last to work with filters)
                loadDraftedState();
            }
        </script>
    </head>
    <body>
        <div class="header">
            <h1>🏈 Fantasy Football Draft Sheet 2025</h1>
            <p>Click on any row to mark as drafted</p>
        </div>

        <div class="legend">
    """

    # Add legend items
    for category, color in colors.items():
        html += f'<span class="legend-item" style="background-color: {color};">{category}</span>'

    html += """
        </div>

        <div class="filters">
            <div class="filter-group">
                <label>Search:</label>
                <input type="text" id="search" class="search-input" placeholder="Player name..." onkeyup="searchPlayers()">
            </div>
            <div class="filter-group">
                <label>Position:</label>
                <button class="filter-btn pos-filter active" onclick="filterPosition('ALL')">ALL</button>
                <button class="filter-btn pos-filter" onclick="filterPosition('QB')">QB</button>
                <button class="filter-btn pos-filter" onclick="filterPosition('RB')">RB</button>
                <button class="filter-btn pos-filter" onclick="filterPosition('WR')">WR</button>
                <button class="filter-btn pos-filter" onclick="filterPosition('TE')">TE</button>
            </div>
            <div class="filter-group">
                <label>Tier:</label>
                <button class="filter-btn tier-filter active" onclick="filterTier('ALL')">ALL</button>
                <button class="filter-btn tier-filter" onclick="filterTier('1')">Tier 1</button>
                <button class="filter-btn tier-filter" onclick="filterTier('2')">Tier 2</button>
                <button class="filter-btn tier-filter" onclick="filterTier('3')">Tier 3+</button>
            </div>
            <div class="filter-group">
                <label>ADP:</label>
                <button class="filter-btn adp-filter active" onclick="filterADP('ALL')">ALL</button>
                <button class="filter-btn adp-filter" onclick="filterADP('HAS_ADP')">Has ADP</button>
                <button class="filter-btn adp-filter" onclick="filterADP('NO_ADP')">No ADP</button>
            </div>
            <button class="filter-btn" onclick="resetFilters()" style="background: #dc3545; color: white; border-color: #dc3545;">Reset</button>
        </div>

        <table>
            <thead>
                <tr>
                    <th class="rank" onclick="sortTable(0)">Rank</th>
                    <th class="player" onclick="sortTable(1)">Player</th>
                    <th class="position" onclick="sortTable(2)">Pos</th>
                    <th class="team" onclick="sortTable(3)">Team</th>
                    <th class="age" onclick="sortTable(4)">Age</th>
                    <th class="points" onclick="sortTable(5)">Proj</th>
                    <th class="ol-rank" onclick="sortTable(6)">OL</th>
                    <th class="adp" onclick="sortTable(7)">ADP</th>
                    <th class="edge" onclick="sortTable(8)">Edge</th>
                    <th class="edge-category" onclick="sortTable(9)">Category</th>
                    <th class="tier" onclick="sortTable(10)">Tier</th>
                    <th class="notes" onclick="sortTable(11)">Notes</th>
                </tr>
            </thead>
            <tbody>
    """

    # Add table rows
    for _, row in df.iterrows():
        rank = int(row['Model_Rank'])

        # Determine row class based on rank
        row_class = ""
        if rank <= 25:
            row_class = "top-25"
        elif rank <= 50:
            row_class = "top-50"
        elif rank <= 100:
            row_class = "top-100"

        # Get colors
        edge_color = colors.get(row['Edge_Category'], '#6c757d')
        pos_color = position_colors.get(row['Position'], '#ffffff')

        # Create tier from position and rank
        pos = row['Position']
        if pos == 'QB':
            if rank <= 12:
                tier = 'QB1'
            elif rank <= 24:
                tier = 'QB2'
            else:
                tier = 'QB3+'
        elif pos == 'RB':
            if rank <= 24:
                tier = 'RB1/2'
            elif rank <= 36:
                tier = 'RB3'
            else:
                tier = 'RB4+'
        elif pos == 'WR':
            if rank <= 24:
                tier = 'WR1/2'
            elif rank <= 36:
                tier = 'WR3'
            else:
                tier = 'WR4+'
        elif pos == 'TE':
            if rank <= 12:
                tier = 'TE2'
            else:
                tier = 'TE3+'
        else:
            tier = 'Other'

        tier_color = tier_colors.get(tier, '#6c757d')

        # Format edge value
        edge_val = row['Rank_Difference']
        if pd.isna(edge_val):
            edge_display = "N/A"
        else:
            edge_display = f"{edge_val:+.0f}" if edge_val != 0 else "0"

        # Create notes from available data
        notes_list = []
        if edge_val and abs(edge_val) >= 20:
            notes_list.append(f"Model {'loves' if edge_val > 0 else 'fades'} ({edge_val:+.0f})")
        if row['Age'] >= 30:
            notes_list.append("Age concern")
        if row['Age'] <= 23:
            notes_list.append("Young upside")
        notes = '<br>'.join(notes_list) if notes_list else ''

        # Get OL rank and ADP, format colors
        ol_rank = row.get('OL_Rank', 'N/A')
        ol_color = '#28a745' if ol_rank != 'N/A' and ol_rank <= 10 else '#ffc107' if ol_rank != 'N/A' and ol_rank <= 20 else '#dc3545' if ol_rank != 'N/A' else '#6c757d'

        adp = row.get('ADP', None)
        adp_display = f"{adp:.1f}" if pd.notna(adp) else '-'
        adp_color = '#28a745' if pd.notna(adp) and adp <= 50 else '#ffc107' if pd.notna(adp) and adp <= 100 else '#dc3545' if pd.notna(adp) else '#6c757d'

        html += f"""
                <tr class="{row_class}" onclick="toggleDrafted(this)">
                    <td class="rank">{rank}</td>
                    <td class="player">{row['Player']}</td>
                    <td class="position" style="background-color: {pos_color};">{row['Position']}</td>
                    <td class="team">{row['Team']}</td>
                    <td class="age">{row['Age']}</td>
                    <td class="points">{row['Model_Points']:.1f}</td>
                    <td class="ol-rank" style="color: {ol_color};">{ol_rank if ol_rank != 'N/A' else '-'}</td>
                    <td class="adp" style="color: {adp_color};">{adp_display}</td>
                    <td class="edge">{edge_display}</td>
                    <td class="edge-category" style="background-color: {edge_color};">{row['Edge_Category']}</td>
                    <td class="tier" style="background-color: {tier_color};">{tier}</td>
                    <td class="notes">{notes}</td>
                </tr>
        """

    html += """
            </tbody>
        </table>
    </body>
    </html>
    """

    # Write HTML file
    with open('draft_sheet_2025.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print("✅ Draft sheet created successfully!")
    print("📄 Open 'draft_sheet_2025.html' in your browser")
    print("\n🎯 Options:")
    print("• python3 format_draft_sheet.py - Create main draft sheet")
    print("• python3 format_draft_sheet.py rookies - Check missing rookies")
    print("• python3 format_draft_sheet.py mobile - Create mobile version")

def check_rookies():
    """Check for missing key 2025 rookies"""
    key_rookies = {
        'RB': ['Ashton Jeanty', 'Omarion Hampton', 'TreVeyon Henderson', 'RJ Harvey'],
        'WR': ['Travis Hunter', 'Tetairoa McMillan', 'Emeka Egbuka', 'Luther Burden III'],
        'QB': ['Shedeur Sanders', 'Cam Ward', 'Quinn Ewers', 'Jalen Milroe'],
        'TE': ['Tyler Warren', 'Colston Loveland', 'Harold Fannin Jr.']
    }

    df = pd.read_csv('draft_sheet_2025.csv')
    missing = []

    for pos, rookies in key_rookies.items():
        for rookie in rookies:
            found = df[df['Player'].str.contains(rookie.split()[-1], case=False, na=False)]
            if found.empty:
                missing.append((pos, rookie))

    if missing:
        print(f"Missing {len(missing)} key rookies:")
        for pos, name in missing:
            print(f"  • {name} ({pos})")
    else:
        print("All key rookies found in projections")

    return missing

def create_mobile_version():
    """Create mobile-optimized version"""
    df = pd.read_csv('draft_sheet_2025.csv')
    team_df = pd.read_csv('../team_data/2025/team_context_2025.csv')
    df = df.merge(team_df[['Team', 'OL_Rank']], on='Team', how='left')

    colors = {'Strong Buy': '#28a745', 'Buy': '#6f9654', 'Consensus': '#6c757d', 'Fade': '#fd7e14', 'Strong Fade': '#dc3545'}

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mobile Draft 2025</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; margin: 10px; background: #f5f5f5; }}
        .player-card {{ background: white; margin: 8px 0; padding: 12px; border-radius: 8px; border-left: 4px solid; }}
        .player-name {{ font-weight: bold; font-size: 16px; }}
        .details {{ font-size: 13px; color: #666; margin: 4px 0; }}
        .drafted {{ opacity: 0.4; text-decoration: line-through; }}
    </style>
    <script>
        function toggleDrafted(card) {{ card.classList.toggle('drafted'); }}
    </script>
</head>
<body>
    <h2>🏈 Draft Sheet 2025</h2>
"""

    for _, row in df.head(100).iterrows():  # Top 100 for mobile
        edge_color = colors.get(row['Edge_Category'], '#6c757d')
        ol_rank = row.get('OL_Rank', 'N/A')

        html += f"""
    <div class="player-card" style="border-left-color: {edge_color};" onclick="toggleDrafted(this)">
        <div class="player-name">{row['Player']}</div>
        <div class="details">{row['Position']} {row['Team']} • {row['Adjusted_Fantasy_Points']:.1f} pts • OL: {ol_rank}</div>
        <div class="details">{row['Edge_Category']} • {row['Value_Tier']}</div>
    </div>"""

    html += "</body></html>"

    with open('mobile_draft_sheet.html', 'w') as f:
        f.write(html)
    print("📱 Mobile version created")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'rookies':
            check_rookies()
        elif sys.argv[1] == 'mobile':
            create_mobile_version()
        else:
            print("Usage: python3 format_draft_sheet.py [rookies|mobile]")
    else:
        format_draft_sheet()
