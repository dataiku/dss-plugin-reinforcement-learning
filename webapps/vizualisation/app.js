const state = {
    runs: [],
    runsById: {},
    selectedRunId: null
};

function toNumber(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function formatNumber(value, digits) {
    if (value === null || value === undefined) {
        return '-';
    }
    const parsed = toNumber(value);
    if (parsed === null) {
        return '-';
    }
    return parsed.toFixed(digits === undefined ? 2 : digits);
}

function formatDate(dateLike) {
    if (!dateLike) {
        return '-';
    }
    const parsedDate = new Date(dateLike);
    if (Number.isNaN(parsedDate.getTime())) {
        return String(dateLike).split('.')[0];
    }
    return parsedDate.toLocaleString();
}

function getRunScores(run) {
    if (!run || !run.data || !Array.isArray(run.data.score)) {
        return [];
    }
    return run.data.score
        .map(function(value) { return toNumber(value); })
        .filter(function(value) { return value !== null; });
}

function runAgent(run) {
    return (run.data && (run.data.agent_name || run.data.agent)) || 'Agent';
}

function runEnvironment(run) {
    return (run.data && (run.data.environmentName || run.data.environment)) || 'Environment';
}

function updateMetric(id, value, digits) {
    $('#' + id).text(formatNumber(value, digits));
}

function populateRunSelector() {
    const select = $('#runSelect');
    select.empty();

    state.runs.forEach(function(run, index) {
        const dateText = formatDate(run.data.trainingdate || run.modified_at);
        const optionText = (index + 1) + '. ' + runAgent(run) + ' | ' + runEnvironment(run) + ' | ' + dateText;
        const option = $('<option></option>').val(run.id).text(optionText);
        select.append(option);
    });
}

function renderComparisonTable() {
    const rows = state.runs.map(function(run) {
        const rowClass = run.id === state.selectedRunId ? ' class="active"' : '';
        return (
            '<tr data-run-id="' + run.id + '"' + rowClass + '>' +
                '<td>' + escapeHtml(run.filename) + '</td>' +
                '<td>' + escapeHtml(runAgent(run)) + '</td>' +
                '<td>' + escapeHtml(runEnvironment(run)) + '</td>' +
                '<td>' + formatNumber(run.summary.average_score, 2) + '</td>' +
                '<td>' + formatNumber(run.summary.best_score, 2) + '</td>' +
                '<td>' + formatNumber(run.summary.episodes, 0) + '</td>' +
                '<td>' + escapeHtml(formatDate(run.data.trainingdate || run.modified_at)) + '</td>' +
            '</tr>'
        );
    });
    $('#comparisonRows').html(rows.join(''));
}

function emptyChart(title) {
    return '<div class="helper-text">' + title + '</div>';
}

function buildLineChart(scores) {
    if (scores.length === 0) {
        return emptyChart('No score series available for this run');
    }

    const width = 820;
    const height = 240;
    const padding = { top: 14, right: 14, bottom: 34, left: 48 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    let minValue = Math.min.apply(null, scores);
    let maxValue = Math.max.apply(null, scores);
    if (minValue === maxValue) {
        minValue -= 1;
        maxValue += 1;
    }
    const range = maxValue - minValue;

    function xPos(index) {
        if (scores.length === 1) {
            return padding.left + plotWidth / 2;
        }
        return padding.left + (index / (scores.length - 1)) * plotWidth;
    }

    function yPos(value) {
        return padding.top + ((maxValue - value) / range) * plotHeight;
    }

    const linePoints = scores.map(function(value, index) {
        return xPos(index).toFixed(2) + ',' + yPos(value).toFixed(2);
    }).join(' ');

    const areaPoints = (
        padding.left + ',' + (padding.top + plotHeight) + ' ' +
        linePoints + ' ' +
        (padding.left + plotWidth) + ',' + (padding.top + plotHeight)
    );

    const gridLines = [];
    for (let i = 0; i <= 4; i += 1) {
        const y = padding.top + (i / 4) * plotHeight;
        gridLines.push('<line x1="' + padding.left + '" y1="' + y + '" x2="' + (padding.left + plotWidth) + '" y2="' + y + '" stroke="#e2e8f1" stroke-width="1"/>');
    }

    const lastIndex = scores.length - 1;
    const lastX = xPos(lastIndex);
    const lastY = yPos(scores[lastIndex]);

    const axisLabels = (
        '<line x1="' + padding.left + '" y1="' + (padding.top + plotHeight) + '" x2="' + (padding.left + plotWidth) + '" y2="' + (padding.top + plotHeight) + '" stroke="#b8c5da" stroke-width="1.2"></line>' +
        '<line x1="' + padding.left + '" y1="' + padding.top + '" x2="' + padding.left + '" y2="' + (padding.top + plotHeight) + '" stroke="#b8c5da" stroke-width="1.2"></line>' +
        '<text x="' + (padding.left + plotWidth / 2) + '" y="' + (height - 8) + '" text-anchor="middle" fill="#5d6b83" font-size="11">Episode</text>' +
        '<text x="14" y="' + (padding.top + plotHeight / 2) + '" text-anchor="middle" fill="#5d6b83" font-size="11" transform="rotate(-90 14 ' + (padding.top + plotHeight / 2) + ')">Score</text>' +
        '<text x="' + padding.left + '" y="' + (height - 18) + '" text-anchor="middle" fill="#7b879c" font-size="10">1</text>' +
        '<text x="' + (padding.left + plotWidth) + '" y="' + (height - 18) + '" text-anchor="middle" fill="#7b879c" font-size="10">' + scores.length + '</text>' +
        '<text x="' + (padding.left - 6) + '" y="' + (padding.top + 10) + '" text-anchor="end" fill="#7b879c" font-size="10">' + formatNumber(maxValue, 2) + '</text>' +
        '<text x="' + (padding.left - 6) + '" y="' + (padding.top + plotHeight) + '" text-anchor="end" fill="#7b879c" font-size="10">' + formatNumber(minValue, 2) + '</text>'
    );

    return (
        '<svg viewBox="0 0 ' + width + ' ' + height + '" preserveAspectRatio="none">' +
            gridLines.join('') +
            axisLabels +
            '<polygon points="' + areaPoints + '" fill="rgba(240,127,71,0.18)"></polygon>' +
            '<polyline points="' + linePoints + '" fill="none" stroke="#f07f47" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"></polyline>' +
            '<circle cx="' + lastX + '" cy="' + lastY + '" r="4.5" fill="#2f8b8f"></circle>' +
        '</svg>'
    );
}

function buildHistogram(scores) {
    if (scores.length === 0) {
        return emptyChart('No score distribution available');
    }

    const width = 420;
    const height = 240;
    const padding = { top: 14, right: 12, bottom: 46, left: 40 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const bins = 8;

    let minValue = Math.min.apply(null, scores);
    let maxValue = Math.max.apply(null, scores);
    if (minValue === maxValue) {
        minValue -= 1;
        maxValue += 1;
    }

    const counts = new Array(bins).fill(0);
    const range = maxValue - minValue;
    scores.forEach(function(score) {
        const normalized = (score - minValue) / range;
        const index = Math.min(bins - 1, Math.floor(normalized * bins));
        counts[index] += 1;
    });

    const maxCount = Math.max.apply(null, counts) || 1;
    const barWidth = plotWidth / bins;
    const axisY = padding.top + plotHeight;
    const bars = counts.map(function(count, index) {
        const barHeight = (count / maxCount) * plotHeight;
        const x = padding.left + index * barWidth + 3;
        const y = padding.top + plotHeight - barHeight;
        return '<rect x="' + x.toFixed(2) + '" y="' + y.toFixed(2) + '" width="' + (barWidth - 6).toFixed(2) + '" height="' + barHeight.toFixed(2) + '" rx="4" fill="#2f8b8f"></rect>';
    }).join('');

    const xTicks = [];
    for (let i = 0; i <= bins; i += 1) {
        const x = padding.left + (i / bins) * plotWidth;
        const value = minValue + (i / bins) * range;
        xTicks.push(
            '<line x1="' + x.toFixed(2) + '" y1="' + axisY.toFixed(2) + '" x2="' + x.toFixed(2) + '" y2="' + (axisY + 4).toFixed(2) + '" stroke="#b8c5da" stroke-width="1"></line>'
        );
        if (i % 2 === 0 || i === bins) {
            xTicks.push(
                '<text x="' + x.toFixed(2) + '" y="' + (axisY + 15).toFixed(2) + '" text-anchor="middle" fill="#7b879c" font-size="9">' + formatNumber(value, 1) + '</text>'
            );
        }
    }

    const axisLabels = (
        '<line x1="' + padding.left + '" y1="' + axisY + '" x2="' + (padding.left + plotWidth) + '" y2="' + axisY + '" stroke="#b8c5da" stroke-width="1.2"></line>' +
        '<line x1="' + padding.left + '" y1="' + padding.top + '" x2="' + padding.left + '" y2="' + (padding.top + plotHeight) + '" stroke="#b8c5da" stroke-width="1.2"></line>' +
        xTicks.join('') +
        '<text x="' + (padding.left + plotWidth / 2) + '" y="' + (height - 4) + '" text-anchor="middle" fill="#5d6b83" font-size="11">Score bins</text>' +
        '<text x="14" y="' + (padding.top + plotHeight / 2) + '" text-anchor="middle" fill="#5d6b83" font-size="11" transform="rotate(-90 14 ' + (padding.top + plotHeight / 2) + ')">Frequency</text>' +
        '<text x="' + (padding.left - 6) + '" y="' + (padding.top + 10) + '" text-anchor="end" fill="#7b879c" font-size="10">' + maxCount + '</text>' +
        '<text x="' + (padding.left - 6) + '" y="' + (padding.top + plotHeight) + '" text-anchor="end" fill="#7b879c" font-size="10">0</text>'
    );

    return (
        '<svg viewBox="0 0 ' + width + ' ' + height + '" preserveAspectRatio="none">' +
            axisLabels +
            bars +
        '</svg>'
    );
}

function findPreviousComparableRun(selectedRun) {
    const selectedIndex = state.runs.findIndex(function(run) { return run.id === selectedRun.id; });
    if (selectedIndex < 0) {
        return null;
    }

    const selectedAgent = runAgent(selectedRun);
    const selectedEnvironment = runEnvironment(selectedRun);
    for (let i = selectedIndex + 1; i < state.runs.length; i += 1) {
        const candidate = state.runs[i];
        if (runAgent(candidate) === selectedAgent && runEnvironment(candidate) === selectedEnvironment) {
            return candidate;
        }
    }
    return null;
}

function renderDelta(run) {
    const valueContainer = $('#deltaScoreId');
    valueContainer.removeClass('delta-positive delta-negative');

    const currentAverage = toNumber(run.summary.average_score);
    const previousRun = findPreviousComparableRun(run);
    if (!previousRun) {
        valueContainer.text('N/A');
        return;
    }

    const previousAverage = toNumber(previousRun.summary.average_score);
    if (currentAverage === null || previousAverage === null) {
        valueContainer.text('N/A');
        return;
    }

    const delta = currentAverage - previousAverage;
    const signedText = (delta >= 0 ? '+' : '') + delta.toFixed(2);
    valueContainer.text(signedText);
    if (delta > 0) {
        valueContainer.addClass('delta-positive');
    } else if (delta < 0) {
        valueContainer.addClass('delta-negative');
    }
}

function renderRun(runId) {
    const run = state.runsById[runId];
    if (!run) {
        return;
    }

    state.selectedRunId = runId;
    $('#runSelect').val(runId);

    const scores = getRunScores(run);
    $('#algorithmId').text(runAgent(run));
    $('#environmentId').text(runEnvironment(run));
    $('#timestampId').text('Training date: ' + formatDate(run.data.trainingdate || run.modified_at));

    updateMetric('averageScoreId', run.summary.average_score, 2);
    updateMetric('bestScoreId', run.summary.best_score, 2);
    updateMetric('worstScoreId', run.summary.worst_score, 2);
    updateMetric('stdScoreId', run.summary.std_score, 2);
    updateMetric('episodeCountId', run.summary.episodes, 0);
    renderDelta(run);

    $('#policyId').text(run.data.policy || '-');
    $('#learningRateId').text(run.data.lr !== undefined ? String(run.data.lr) : '-');
    $('#gammaId').text(run.data.gamma !== undefined ? String(run.data.gamma) : '-');
    $('#nbTrainingEpisodesId').text(run.data.total_timesteps || run.data.total_episodes || '-');

    $('#scoreTrendChart').html(buildLineChart(scores));
    $('#scoreDistributionChart').html(buildHistogram(scores));
    $('#scoreTrendMeta').text(scores.length > 0 ? (scores.length + ' episodes, last score ' + formatNumber(scores[scores.length - 1], 2)) : '');

    renderComparisonTable();
}

function renderError(message, details) {
    let fullMessage = message;
    if (Array.isArray(details) && details.length > 0) {
        const first = details[0];
        const file = first.file ? (' [' + first.file + ']') : '';
        const reason = first.error ? (': ' + first.error) : '';
        fullMessage = message + file + reason;
    }
    $('#loadError').text(fullMessage).removeClass('hidden');
    $('#algorithmId').text('No results');
    $('#environmentId').text('available');
    $('#timestampId').text(fullMessage);
}

function renderManifestError(message) {
    $('#manifestMeta').text('Manifest helper unavailable');
    $('#manifestError').text(message).removeClass('hidden');
    $('#manifestProfiles').html('');
    $('#profileSlugsOutput').val('');
}

function renderManifestHelper(payload) {
    if (!payload || payload.error) {
        renderManifestError(payload && payload.error ? payload.error : 'Manifest lookup failed');
        return;
    }

    $('#manifestError').addClass('hidden').text('');
    const manifest = payload.manifest || {};
    const profiles = Array.isArray(manifest.profiles) ? manifest.profiles : [];
    const slugs = profiles
        .map(function(profile) { return profile.profile_slug; })
        .filter(function(slug) { return !!slug; });

    if (profiles.length === 0) {
        renderManifestError('No profiles found in manifest');
        return;
    }

    const profilePills = profiles.map(function(profile) {
        const slug = profile.profile_slug || 'unknown';
        const name = profile.profile_name || slug;
        const agent = profile.agent || '-';
        const environment = profile.environment || '-';
        return (
            '<span class="manifest-pill" title="' + escapeHtml(agent + ' | ' + environment) + '">' +
                escapeHtml(name) + ' (' + escapeHtml(slug) + ')' +
            '</span>'
        );
    });

    const manifestMeta = (
        profiles.length + ' profile(s) from ' +
        (payload.source || 'manifest source') +
        (manifest.generated_at ? (' | generated ' + formatDate(manifest.generated_at)) : '')
    );
    $('#manifestMeta').text(manifestMeta);
    $('#manifestProfiles').html(profilePills.join(''));
    $('#profileSlugsOutput').val(slugs.join(',\n'));
}

function copyProfileSlugsToClipboard() {
    const value = $('#profileSlugsOutput').val() || '';
    if (!value.trim()) {
        return;
    }
    const button = $('#copyProfileSlugsBtn');
    const previousText = button.text();

    function setCopiedState(text) {
        button.text(text);
        setTimeout(function() {
            button.text(previousText);
        }, 1400);
    }

    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(value)
            .then(function() { setCopiedState('Copied'); })
            .catch(function() { setCopiedState('Copy failed'); });
        return;
    }

    $('#profileSlugsOutput').trigger('focus').trigger('select');
    try {
        const copied = document.execCommand('copy');
        setCopiedState(copied ? 'Copied' : 'Copy failed');
    } catch (err) {
        setCopiedState('Copy failed');
    }
}

function bootstrapDashboard(payload) {
    $('#loadError').addClass('hidden').text('');
    state.runs = Array.isArray(payload.runs) ? payload.runs : [];
    state.runsById = {};
    state.runs.forEach(function(run) {
        state.runsById[run.id] = run;
    });

    if (state.runs.length === 0) {
        renderError('No valid run metrics found');
        return;
    }

    const videoText = payload.video_count > 0 ? (' | ' + payload.video_count + ' video file(s) in folder') : '';
    const parseErrors = Array.isArray(payload.errors) && payload.errors.length > 0 ? (' | ' + payload.errors.length + ' file(s) skipped') : '';
    $('#runCountLabel').text(state.runs.length + ' run(s) loaded' + videoText + parseErrors);

    populateRunSelector();
    const initialRunId = payload.latest_run_id || state.runs[0].id;
    renderRun(initialRunId);
}

$('#runSelect').on('change', function() {
    renderRun($(this).val());
});

$('#comparisonRows').on('click', 'tr', function() {
    const runId = $(this).attr('data-run-id');
    renderRun(runId);
});

$('#copyProfileSlugsBtn').on('click', function() {
    copyProfileSlugsToClipboard();
});

$.getJSON(getWebAppBackendUrl('/first_api_call'))
    .done(function(payload) {
        if (payload.error) {
            renderError(payload.error, payload.errors);
            return;
        }
        bootstrapDashboard(payload);
    })
    .fail(function() {
        renderError('Backend request failed');
    });

$.getJSON(getWebAppBackendUrl('/manifest_profiles'))
    .done(function(payload) {
        renderManifestHelper(payload);
    })
    .fail(function() {
        renderManifestError('Manifest endpoint request failed');
    });
