document.addEventListener('DOMContentLoaded', () => {
    // Inject the IEEE Toggle button into the global navigation
    const navs = document.querySelectorAll('.global-nav');

    if (navs.length > 0) {
        navs.forEach(nav => {
            const toggleBtn = document.createElement('button');
            toggleBtn.id = 'ieeeToggleBtn';
            toggleBtn.className = 'nav-link';
            toggleBtn.style.marginLeft = '10px';
            toggleBtn.style.cursor = 'pointer';
            toggleBtn.style.fontWeight = 'bold';
            toggleBtn.style.border = '1px solid var(--border)';
            toggleBtn.textContent = 'IEEE Style';

            nav.appendChild(toggleBtn);

            toggleBtn.addEventListener('click', (e) => {
                e.preventDefault();
                toggleIeeeTheme();
            });
        });
    }

    // Initialize theme based on localStorage
    if (localStorage.getItem('ieee-theme') === 'true') {
        document.documentElement.setAttribute('data-theme', 'ieee');
        applyChartDefaults(true);
        updateToggleButton(true);
    }
});

function applyChartDefaults(isIeee) {
    if (typeof Chart === 'undefined') return;
    const textColor = isIeee ? '#111111' : '#94a3b8';
    const gridColor = isIeee ? 'rgba(0,0,0,0.15)' : 'rgba(255,255,255,0.08)';
    Chart.defaults.color = textColor;
    Chart.defaults.scale.grid.color = gridColor;
    Chart.defaults.scale.ticks.color = textColor;
    Chart.defaults.plugins.legend.labels.color = textColor;
    // Re-render all existing chart instances
    if (Chart.instances) {
        Object.values(Chart.instances).forEach(chart => {
            if (chart && chart.options) {
                ['x', 'y', 'r'].forEach(axis => {
                    if (chart.options.scales && chart.options.scales[axis]) {
                        if (chart.options.scales[axis].ticks) {
                            chart.options.scales[axis].ticks.color = textColor;
                        }
                        if (chart.options.scales[axis].title) {
                            chart.options.scales[axis].title.color = textColor;
                        }
                        if (chart.options.scales[axis].grid) {
                            chart.options.scales[axis].grid.color = gridColor;
                        }
                    }
                });
                chart.update('none');
            }
        });
    }
}

function toggleIeeeTheme() {
    const isIeee = document.documentElement.getAttribute('data-theme') === 'ieee';

    if (isIeee) {
        document.documentElement.removeAttribute('data-theme');
        localStorage.setItem('ieee-theme', 'false');
        applyChartDefaults(false);
        updateToggleButton(false);
    } else {
        document.documentElement.setAttribute('data-theme', 'ieee');
        localStorage.setItem('ieee-theme', 'true');
        applyChartDefaults(true);
        updateToggleButton(true);
    }
}

function updateToggleButton(isIeee) {
    const btns = document.querySelectorAll('#ieeeToggleBtn');
    btns.forEach(btn => {
        if (isIeee) {
            btn.textContent = 'Standard Style';
            btn.classList.add('active');
        } else {
            btn.textContent = 'IEEE Style';
            btn.classList.remove('active');
        }
    });
}
