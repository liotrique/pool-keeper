document.addEventListener('DOMContentLoaded', () => {
    let playersData = [];
    let draftState = JSON.parse(localStorage.getItem('draftState')) || {}; 
    // draftState format: { "Macklin Celebrini": "MINE", "Connor McDavid": "OTHER" }
    
    let currentSort = { column: 'Score Valeur Dynastie', desc: true };

    // Éléments du DOM
    const tbody = document.getElementById('players-body');
    const searchInput = document.getElementById('search-input');
    const posFilter = document.getElementById('pos-filter');
    const statusFilter = document.getElementById('status-filter');
    const resetBtn = document.getElementById('reset-draft');
    
    // Chargement du CSV
    Papa.parse('../predictions_keeper_20262027.csv', {
        download: true,
        header: true,
        dynamicTyping: true,
        complete: function(results) {
            playersData = results.data.filter(p => p.Nom); // Filtrer les lignes vides
            renderApp();
        },
        error: function(err) {
            console.error("Erreur de chargement du CSV:", err);
            tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; color: var(--danger);">Erreur de chargement du fichier CSV. Assurez-vous qu'il se nomme bien 'predictions_keeper_20262027.csv' et se trouve dans le dossier parent.</td></tr>`;
        }
    });

    // Écouteurs d'événements
    searchInput.addEventListener('input', renderApp);
    posFilter.addEventListener('change', renderApp);
    statusFilter.addEventListener('change', renderApp);
    resetBtn.addEventListener('click', () => {
        if (confirm("Voulez-vous vraiment réinitialiser le repêchage ?")) {
            draftState = {};
            saveState();
            renderApp();
        }
    });

    // Tri
    document.querySelectorAll('th[data-sort]').forEach(th => {
        th.addEventListener('click', () => {
            const column = th.dataset.sort;
            if (currentSort.column === column) {
                currentSort.desc = !currentSort.desc;
            } else {
                currentSort.column = column;
                currentSort.desc = true;
            }
            
            // Mise à jour visuelle des headers
            document.querySelectorAll('th').forEach(h => h.classList.remove('sort-active', 'sort-desc', 'sort-asc'));
            th.classList.add('sort-active');
            th.classList.add(currentSort.desc ? 'sort-desc' : 'sort-asc');
            
            renderApp();
        });
    });

    function saveState() {
        localStorage.setItem('draftState', JSON.stringify(draftState));
    }

    function setPlayerStatus(name, status) {
        if (status === 'AVAILABLE') {
            delete draftState[name];
        } else {
            draftState[name] = status;
        }
        saveState();
        renderApp();
    }

    // Expose fonction pour les boutons
    window.actionDraft = function(name, status) {
        setPlayerStatus(name, status);
    };

    function renderApp() {
        renderTable();
        renderRoster();
    }

    function renderTable() {
        const query = searchInput.value.toLowerCase();
        const pos = posFilter.value;
        const status = statusFilter.value;

        // Filtrage
        let filtered = playersData.filter(p => {
            if (query && !p.Nom.toLowerCase().includes(query)) return false;
            
            // Les attaquants regroupent C, L, R. Si on filtre spécifique, on check exactement.
            if (pos !== 'ALL') {
                if (pos === 'C' || pos === 'L' || pos === 'R') {
                    if (p.Position !== pos) return false;
                } else if (pos === 'D' && p.Position !== 'D') return false;
                else if (pos === 'G' && p.Position !== 'G') return false;
            }

            const pStatus = draftState[p.Nom] || 'AVAILABLE';
            if (status !== 'ALL' && status !== pStatus) return false;

            return true;
        });

        // Tri
        filtered.sort((a, b) => {
            let valA = a[currentSort.column];
            let valB = b[currentSort.column];
            
            if (typeof valA === 'string') valA = valA.toLowerCase();
            if (typeof valB === 'string') valB = valB.toLowerCase();
            
            if (valA < valB) return currentSort.desc ? 1 : -1;
            if (valA > valB) return currentSort.desc ? -1 : 1;
            return 0;
        });

        // Rendu
        tbody.innerHTML = '';
        filtered.forEach(p => {
            const pStatus = draftState[p.Nom] || 'AVAILABLE';
            const tr = document.createElement('tr');
            
            if (pStatus === 'MINE') tr.classList.add('status-mine');
            if (pStatus === 'OTHER') tr.classList.add('status-other');

            // Format de la position
            let posClass = 'pos-tag ';
            if (['C','L','R'].includes(p.Position)) posClass += 'pos-att';
            else if (p.Position === 'D') posClass += 'pos-def';
            else if (p.Position === 'G') posClass += 'pos-gar';

            const pts24 = (p['Points Pool 2024-25'] || 0).toFixed(1);
            const pts25 = (p['Points Pool 2025-26'] || 0).toFixed(1);
            const pred = (p['Prédiction Brute 2026-27'] || 0).toFixed(1);
            const dyn = (p['Score Valeur Dynastie'] || 0).toFixed(1);
            const age = (p['Âge'] || 0).toFixed(1);

            let actionBtns = '';
            if (pStatus === 'AVAILABLE') {
                actionBtns = `
                    <div class="action-btns">
                        <button class="btn-mine" onclick="actionDraft('${p.Nom.replace(/'/g, "\\'")}', 'MINE')">+ Mon Équipe</button>
                        <button class="btn-other" onclick="actionDraft('${p.Nom.replace(/'/g, "\\'")}', 'OTHER')">Pris</button>
                    </div>
                `;
            } else {
                actionBtns = `
                    <button class="btn-other" onclick="actionDraft('${p.Nom.replace(/'/g, "\\'")}', 'AVAILABLE')">Annuler</button>
                `;
            }

            tr.innerHTML = `
                <td><strong>${p.Nom}</strong></td>
                <td><span class="${posClass}" style="background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px;">${p.Position}</span></td>
                <td>${age}</td>
                <td class="hide-mobile">${pts24}</td>
                <td class="hide-mobile">${pts25}</td>
                <td>${pred}</td>
                <td class="score-highlight">${dyn}</td>
                <td>${actionBtns}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    function renderRoster() {
        const rosterAtt = document.getElementById('roster-attaquants');
        const rosterDef = document.getElementById('roster-defenseurs');
        const rosterGar = document.getElementById('roster-gardiens');
        
        rosterAtt.innerHTML = '';
        rosterDef.innerHTML = '';
        rosterGar.innerHTML = '';

        let countAtt = 0;
        let countDef = 0;
        let countGar = 0;

        playersData.forEach(p => {
            if (draftState[p.Nom] === 'MINE') {
                const li = document.createElement('li');
                li.className = 'roster-item';
                li.innerHTML = `
                    <div class="info">
                        <span class="name">${p.Nom} <span style="font-size: 0.7rem; color: var(--accent-cyan);">(${p.Position})</span></span>
                        <span class="stats">Dynastie: ${p['Score Valeur Dynastie'].toFixed(1)}</span>
                    </div>
                    <button class="btn-remove" onclick="actionDraft('${p.Nom.replace(/'/g, "\\'")}', 'AVAILABLE')">×</button>
                `;

                if (['C', 'L', 'R'].includes(p.Position)) {
                    rosterAtt.appendChild(li);
                    countAtt++;
                } else if (p.Position === 'D') {
                    rosterDef.appendChild(li);
                    countDef++;
                } else if (p.Position === 'G') {
                    rosterGar.appendChild(li);
                    countGar++;
                }
            }
        });

        document.getElementById('count-att').textContent = `${countAtt}/12`;
        document.getElementById('count-def').textContent = `${countDef}/6`;
        document.getElementById('count-gar').textContent = `${countGar}/2`;
        
        document.getElementById('count-att').style.color = countAtt > 12 ? 'var(--danger)' : 'var(--accent-cyan)';
        document.getElementById('count-def').style.color = countDef > 6 ? 'var(--danger)' : 'var(--accent-cyan)';
        document.getElementById('count-gar').style.color = countGar > 2 ? 'var(--danger)' : 'var(--accent-cyan)';
    }
});
